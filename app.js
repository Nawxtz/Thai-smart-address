/* ThaiSmartAddress v7.0 — Dashboard application logic */
/* Extracted from index.html to satisfy CSP script-src 'self' */

/* ── Sample data ──────────────────────────────────────────── */
const SAMPLES = [
  "รบกวนส่งที่ คุณแม็ค 99/9 หมู่บ้านสุขสันต์ ตำบลแสนสุข อำเภอเมืองชลบุรี จังหวัดชลบุรี 20130 0812345678 ด่วน",
  "ส่งนายสมชาย 081-999-8888 ท่าแร้ง บางเขน กทม 10220 ระวังของแตก",
  "โอนแล้วค่ะ ส่งที่ น้องแนน 100/1 หมู่ 7 ช้างเผือก เชียงใหม่ 50300 0921234567",
  "ส่งให้ด้วยนะคะ โทร 0812222333",
  "99/9 สุเทพ เมืองเชียงใหม่ เชียงใหม่ 50200 064-111-2222 ฝากป้อมยาม คุณดวง ระวังของแตก ด่วนมาก",
];

function loadSample(i) {
  document.getElementById('chat-input').value = SAMPLES[i];
  document.getElementById('chat-input').focus();
}

/* ── API base ─────────────────────────────────────────────── */
function getBase() {
  const v = document.getElementById('api-base').value.trim();
  return v.replace(/\/$/, '');
}

/* FIX [#4]: Read API key from the input field and include it in all requests.
   Without this, every request returns 401 when API_KEY is configured. */
function getApiKey() {
  return document.getElementById('api-key-input')?.value.trim() || '';
}

function authHeaders(extra) {
  const h = { 'Content-Type': 'application/json', ...extra };
  const key = getApiKey();
  if (key) h['X-API-Key'] = key;
  return h;
}

/* ── Connection test ─────────────────────────────────────── */
async function testConnection() {
  const dot  = document.getElementById('api-dot');
  const txt  = document.getElementById('api-status-text');
  const disp = document.getElementById('api-url-display');
  const btn  = document.querySelector('.btn-test');
  txt.textContent = 'กำลังทดสอบ…';
  dot.className   = 'status-dot';
  // FIX BUG-NEW-5: Replace AbortSignal.timeout(4000) with AbortController.
  // AbortSignal.timeout() is unavailable in older browsers (pre-Chrome 103,
  // pre-Firefox 100, pre-Safari 16) and throws TypeError, preventing the
  // catch block from re-enabling the Test button — UI freezes permanently.
  if (btn) btn.disabled = true;
  const _ctrl    = new AbortController();
  const _timeout = setTimeout(() => _ctrl.abort(), 4000);
  try {
    const r  = await fetch(`${getBase()}/api/health`, { signal: _ctrl.signal });
    clearTimeout(_timeout);
    const d  = await r.json();
    dot.classList.add('online');
    txt.textContent = `Online — ${d.geo_records} geo records`;
    disp.textContent = getBase();
  } catch (e) {
    clearTimeout(_timeout);
    dot.className   = 'status-dot';
    txt.textContent = `ไม่สามารถเชื่อมต่อ (${e.message})`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* ── Parse ────────────────────────────────────────────────── */
async function parseAddress() {
  const text = document.getElementById('chat-input').value.trim();
  if (!text) {
    alert('กรุณาใส่ข้อความที่อยู่ก่อนค่ะ');
    return;
  }

  const btn = document.getElementById('btn-parse');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>กำลังประมวลผล…';

  // Hide empty state
  document.getElementById('empty-state').style.display = 'none';

  let data;
  // FIX [Bug 2 frontend]: AbortSignal.timeout() is not supported in all browsers
  // (requires Chrome 103+, Firefox 100+, Safari 16+).  Use a manual AbortController
  // with setTimeout as a universal fallback so the UI ALWAYS recovers within 12 s
  // even if the server's event loop is blocked (old backend) or the network is slow.
  const _ctrl    = new AbortController();
  const _timeout = setTimeout(() => _ctrl.abort(), 12000);
  try {
    const res  = await fetch(`${getBase()}/api/parse`, {
      method:  'POST',
      headers: authHeaders(),
      body:    JSON.stringify({ text }),
      signal:  _ctrl.signal,
    });
    clearTimeout(_timeout);
    data = await res.json();
    if (!res.ok) {
      data = { status: 'Error', confidence: 0, warnings: [JSON.stringify(data)], request_id: '—' };
    }
  } catch (e) {
    clearTimeout(_timeout);
    const msg = e.name === 'AbortError'
      ? 'คำขอใช้เวลานานเกิน 12 วินาที — เซิร์ฟเวอร์อาจโหลดหนัก กรุณาลองใหม่'
      : `Network error: ${e.message}`;
    data = { status: 'Error', confidence: 0, warnings: [msg], request_id: '—' };
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ ประมวลผลที่อยู่';
  }

  renderCard(text, data);
  addHistory(data.status);
}

/* ── Render result card ───────────────────────────────────── */
// FIX BUG-NEW-11: Bounded card registry — limit to last 20 cards.
// Previous code grew _cardRegistry indefinitely, keeping all parsed results
// and their DOM nodes in memory for the entire session.
const _cardRegistry = new Map();
const _MAX_CARDS = 20;

// FIX BUG-NEW-12: Monotonic counter for card IDs prevents millisecond collision.
// Date.now() has 1ms resolution — a double-click produces two cards with the
// same ID; the second _cardRegistry.set() overwrites the first, then
// submitFeedback('card-X') sends the wrong card's data for the first card.
let _cardSeq = 0;

function renderCard(originalText, data) {
  const panel   = document.getElementById('results-panel');
  const isEmpty = document.getElementById('empty-state');
  if (isEmpty) isEmpty.remove();

  // Determine CSS class
  const statusClass =
    data.status === 'Success' || data.status === 'Success with Warnings'
      ? 'status-success'
      : data.status === 'Flagged for Review'
      ? 'status-flagged'
      : 'status-error';

  const confPct  = Math.round((data.confidence || 0) * 100);
  const latency  = data.processing_ms != null ? `${data.processing_ms} ms` : '—';
  const reqId    = data.request_id || '—';

  // Warnings HTML
  let warningsHtml = '';
  if (data.warnings && data.warnings.length && statusClass !== 'status-success') {
    warningsHtml = `
      <div class="warnings-block">
        ${data.warnings.map(w => `
          <div class="warning-row">
            <div class="warning-icon">⚠</div>
            <span>${escHtml(w)}</span>
          </div>`).join('')}
      </div>`;
  }

  // Tags HTML
  let tagsHtml = '';
  if (data.tags && data.tags.length) {
    tagsHtml = `
      <div class="tags-block">
        ${data.tags.map(t => `<span class="tag-chip ${escHtml(t)}">${tagLabel(t)}</span>`).join('')}
      </div>`;
  }

  // FIX BUG-NEW-12: Monotonic counter avoids millisecond ID collision.
  const cardId = `card-${++_cardSeq}`;

  const html = `
    <div class="result-card ${statusClass}" id="${cardId}">

      <!-- Header -->
      <div class="card-header ${statusClass}">
        <div class="card-status-badge">
          <div class="badge-dot"></div>
          <span class="badge-text">${escHtml(data.status || 'Error')}</span>
        </div>
        <div class="card-meta">
          <div class="confidence-bar-wrap">
            <span class="confidence-label">CONF</span>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width:${confPct}%"></div>
            </div>
            <span class="confidence-pct">${confPct}%</span>
          </div>
          <span class="latency-badge">${latency}</span>
        </div>
      </div>

      ${warningsHtml}
      ${tagsHtml}

      <!-- Editable Fields -->
      <div class="fields-grid">

        <div class="field-cell full-width">
          <div class="field-label">ชื่อผู้รับ / RECEIVER</div>
          <input class="field-input" id="${cardId}-receiver"
                 value="${escAttr(data.receiver)}" placeholder="ไม่พบข้อมูล" />
        </div>

        <div class="field-cell">
          <div class="field-label">เบอร์โทร / PHONE</div>
          <input class="field-input mono" id="${cardId}-phone"
                 value="${escAttr(data.phone)}" placeholder="—" />
        </div>

        <div class="field-cell">
          <div class="field-label">รหัสไปรษณีย์ / ZIPCODE</div>
          <input class="field-input mono" id="${cardId}-zipcode"
                 value="${escAttr(data.zipcode)}" placeholder="—" />
        </div>

        <div class="field-cell full-width">
          <div class="field-label">รายละเอียดที่อยู่ / ADDRESS DETAIL</div>
          <input class="field-input" id="${cardId}-address_detail"
                 value="${escAttr(data.address_detail)}" placeholder="บ้านเลขที่, ซอย, ถนน, หมู่บ้าน" />
        </div>

        <div class="field-cell">
          <div class="field-label">ตำบล / SUB-DISTRICT</div>
          <input class="field-input" id="${cardId}-sub_district"
                 value="${escAttr(data.sub_district)}" placeholder="—" />
        </div>

        <div class="field-cell">
          <div class="field-label">อำเภอ / DISTRICT</div>
          <input class="field-input" id="${cardId}-district"
                 value="${escAttr(data.district)}" placeholder="—" />
        </div>

        <div class="field-cell full-width">
          <div class="field-label">จังหวัด / PROVINCE</div>
          <input class="field-input" id="${cardId}-province"
                 value="${escAttr(data.province)}" placeholder="—" />
        </div>

      </div>

      <!-- Footer -->
      <div class="card-footer">
        <div class="request-id">
          ID&nbsp;${escHtml(reqId)}
        </div>
        <div style="display:flex;align-items:center;gap:10px">
          <div class="feedback-toast" id="${cardId}-toast">
            ✓ บันทึกแล้ว
          </div>
          <button class="btn-feedback" id="${cardId}-fbtn"
                  onclick="submitFeedback('${cardId}')">
            💾 Save &amp; Submit Correction
          </button>
        </div>
      </div>

    </div>`;

  // Prepend new card (newest on top)
  panel.insertAdjacentHTML('afterbegin', html);
  // FIX [Bug 3]: Store card payload in registry so onclick doesn't need JSON in attributes
  _cardRegistry.set(cardId, { originalText, data });

  // FIX BUG-NEW-11: Prune oldest cards beyond _MAX_CARDS to prevent unbounded
  // memory growth. Remove both the registry entry and the DOM node.
  if (_cardRegistry.size > _MAX_CARDS) {
    // The oldest key is the first key inserted (Map preserves insertion order)
    const oldestId = _cardRegistry.keys().next().value;
    _cardRegistry.delete(oldestId);
    const oldDom = document.getElementById(oldestId);
    if (oldDom) oldDom.remove();
  }
}

/* ── Submit feedback ─────────────────────────────────────── */
// FIX [Bug 3]: submitFeedback now takes only cardId.
// Previously it received JSON.stringify(JSON.stringify(data)) as an HTML attribute,
// which ALWAYS broke on the outer double-quote produced by JSON.stringify — the
// HTML attribute value was terminated at the first " character, silently truncating
// the argument and causing a JS parse error when the button was clicked.
async function submitFeedback(cardId) {
  const entry = _cardRegistry.get(cardId);
  if (!entry) { alert('ไม่พบข้อมูลการ์ด กรุณารีเฟรชและลองใหม่'); return; }
  const { originalText, data: parsedData } = entry;

  const btn   = document.getElementById(`${cardId}-fbtn`);
  const toast = document.getElementById(`${cardId}-toast`);

  // Read current (possibly edited) field values
  const read = id => (document.getElementById(`${cardId}-${id}`)?.value || '').trim() || null;

  // FIX BUG-NEW-13: Spread only address fields into correctedOutput.
  // Previously "...parsedData" copied request_id, processing_ms, confidence,
  // warnings, tags, and status into corrected_output — system metadata that
  // corrupts downstream ML training data. Now we only include address fields.
  const correctedOutput = {
    receiver:       read('receiver'),
    phone:          read('phone'),
    address_detail: read('address_detail'),
    sub_district:   read('sub_district'),
    district:       read('district'),
    province:       read('province'),
    zipcode:        read('zipcode'),
  };

  btn.disabled = true;
  btn.textContent = 'กำลังส่ง…';

  try {
    const res = await fetch(`${getBase()}/api/feedback`, {
      method:  'POST',
      headers: authHeaders(),
      body:    JSON.stringify({
        original_text:    originalText,
        parsed_output:    parsedData,
        corrected_output: correctedOutput,
        corrected_by:     'admin',
        // FIX [Bug 3]: Send request_id (the canonical tracing field) + session_id for
        // backward compat.  Both are now accepted by the updated FeedbackRequest schema.
        request_id:       parsedData.request_id || null,
        session_id:       parsedData.request_id || null,
      }),
      // FIX [Bug 2 frontend]: Use AbortController for universal browser compatibility
      signal: (() => { const c = new AbortController(); setTimeout(() => c.abort(), 8000); return c.signal; })(),
    });

    if (res.ok) {
      toast.style.display = 'flex';
      btn.textContent     = '✓ ส่งแล้ว';
      setTimeout(() => { toast.style.display = 'none'; }, 3500);
    } else {
      const err = await res.json().catch(() => ({}));
      alert(`เกิดข้อผิดพลาด: ${err.detail || res.status}`);
      btn.disabled    = false;
      btn.textContent = '💾 Save & Submit Correction';
    }
  } catch (e) {
    alert(`Network error: ${e.message}`);
    btn.disabled    = false;
    btn.textContent = '💾 Save & Submit Correction';
  }
}

/* ── History strip ───────────────────────────────────────── */
const _history = [];
function addHistory(statusStr) {
  const cls = statusStr === 'Success' || statusStr === 'Success with Warnings' ? 's'
            : statusStr === 'Flagged for Review' ? 'f' : 'e';
  _history.unshift({ cls, label: statusStr });
  const strip = document.getElementById('history-strip');
  strip.innerHTML = _history.slice(0, 30).map(h =>
    `<div class="history-dot ${h.cls}" title="${escHtml(h.label)}"></div>`
  ).join('');
}

/* ── Utils ───────────────────────────────────────────────── */
function escHtml(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escAttr(s) {
  if (s == null) return '';
  return String(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function tagLabel(t) {
  const map = {
    Urgent: '🔴 ด่วน', Fragile: '🔵 แตกง่าย',
    Drop_at_guard: '🟣 ป้อมยาม', Do_not_fold: '🟢 ห้ามพับ', Keep_dry: '🔵 ห้ามเปียก',
  };
  return map[t] || t;
}

/* ── Auto-test connection on load ─────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
  // Restore saved API key (convenience only — not a security control)
  const savedKey = localStorage.getItem('tsa_api_key');
  if (savedKey) {
    const el = document.getElementById('api-key-input');
    if (el) el.value = savedKey;
  }
  document.getElementById('api-url-display').textContent = getBase();
  testConnection();
});

document.addEventListener('input', e => {
  if (e.target && e.target.id === 'api-key-input') {
    localStorage.setItem('tsa_api_key', e.target.value);
  }
});
