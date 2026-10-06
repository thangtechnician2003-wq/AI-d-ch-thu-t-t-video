import streamlit as st
import re
import time
import json
import html
import urllib.request
from datetime import datetime

# Import các thư viện AI
import google.generativeai as genai
from openai import OpenAI
import anthropic

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(
    page_title="Studio Kịch Bản & Video AI", 
    layout="wide", 
    page_icon="🎬"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 25px;
        border: 1px solid #334155;
    }
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
    }
    .main-header p {
        margin: 6px 0 0 0;
        color: #94a3b8;
        font-size: 14px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo bộ nhớ tạm
if "history" not in st.session_state:
    st.session_state.history = []
if "active_id" not in st.session_state:
    st.session_state.active_id = None
if "input_source_text" not in st.session_state:
    st.session_state.input_source_text = ""
if "detected_sub_lang" not in st.session_state:
    st.session_state.detected_sub_lang = ""
if "groq_models" not in st.session_state:
    st.session_state.groq_models = [
        "openai/gpt-oss-120b",
        "deepseek-r1-distill-llama-70b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3.8-27b"
    ]

LANGUAGES = [
    "Tiếng Việt", "Tiếng Anh", "Tiếng Ả Rập", "Tiếng Croatia", "Tiếng Serbia", "Tiếng Ba Lan",
    "Tiếng Séc", "Tiếng Pháp", "Tiếng Đức", "Tiếng Tây Ban Nha", "Tiếng Bồ Đào Nha",
    "Tiếng Hà Lan", "Tiếng Thái Lan", "Tiếng Ấn Độ", "Tiếng Đan Mạch", "Tiếng Na Uy",
    "Tiếng Hungary", "Tiếng Slovenia", "Tiếng Romania", "Tiếng Phần Lan", "Tiếng Thụy Điển",
    "Tiếng Ý", "Tiếng Đức (Áo)", "Tiếng Ireland", "Tiếng Hy Lạp", "Tiếng Hàn",
    "Tiếng Nhật", "Tiếng Pháp (Canada)"
]

# --- THUẬT TOÁN CHIA NHỎ VĂN BẢN (NÂNG LÊN SÁT 90% TRẦN MODEL: 18.000 KÝ TỰ) ---
def split_text_smartly(text, max_chars=18000):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    
    # Ưu tiên tách theo đoạn văn lớn (\n\n) hoặc từng dòng thoại (\n)
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0
    
    for p in paragraphs:
        p_len = len(p) + 1
        if current_len + p_len > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = [p]
            current_len = p_len
        else:
            current_chunk.append(p)
            current_len += p_len
            
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())
        
    final_chunks = []
    for c in chunks:
        if len(c) <= max_chars:
            final_chunks.append(c)
        else:
            sentences = re.split(r'([.?!]\s+)', c)
            sub_chunk = ""
            for s in sentences:
                if len(sub_chunk) + len(s) > max_chars and sub_chunk:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = s
                else:
                    sub_chunk += s
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())
                
    return [c for c in final_chunks if c.strip()]

# --- HÀM TRÍCH XUẤT ID YOUTUBE ---
def extract_video_id(url):
    url = url.strip()
    if "v=" in url:
        return url.split("v=")[1].split("&")[0][:11]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0][:11]
    elif "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0][:11]
    match = re.search(r"([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# --- BỘ GIẢI MÃ PHỤ ĐỀ YOUTUBE ---
def fetch_youtube_subtitles_cloud(video_id):
    try:
        api_url = "https://www.youtube.com/youtubei/v1/player"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "com.google.android.youtube/19.29.35 (Linux; U; Android 11) gzip"
        }
        payload = {
            "context": {
                "client": {
                    "hl": "vi", "gl": "VN",
                    "clientName": "ANDROID", "clientVersion": "19.29.35",
                    "androidSdkVersion": 30
                }
            },
            "videoId": video_id
        }
        req = urllib.request.Request(api_url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            player_data = json.loads(resp.read().decode("utf-8"))
            
        caption_tracks = player_data.get("captions", {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
        if caption_tracks:
            sorted_tracks = sorted(caption_tracks, key=lambda x: 1 if x.get("kind") == "asr" else 0)
            target_track = sorted_tracks[0]
            base_url = target_track.get("baseUrl")
            
            sub_req = urllib.request.Request(base_url + "&fmt=json3", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(sub_req, timeout=10) as s_resp:
                s_json = json.loads(s_resp.read().decode("utf-8"))
                
            lines = []
            for ev in s_json.get("events", []):
                segs = ev.get("segs", [])
                t_line = "".join(s.get("utf8", "") for s in segs if "utf8" in s).strip()
                if t_line and not (t_line.startswith("[") and t_line.endswith("]")):
                    lines.append(t_line)
                    
            if lines:
                name_run = target_track.get("name", {}).get("runs", [{}])[0].get("text")
                lang_name = name_run or target_track.get("name", {}).get("simpleText", target_track.get("languageCode", "Gốc"))
                kind_str = "Tự động tạo (ASR)" if target_track.get("kind") == "asr" else "Phụ đề gốc tác giả"
                return "\n".join(lines), f"{lang_name} [{target_track.get('languageCode', '')}] • {kind_str}", None
    except Exception:
        pass

    instances = [
        "https://inv.nadeko.net",
        "https://invidious.nerdvpn.de",
        "https://invidious.drgns.space",
        "https://yt.artemislena.eu"
    ]
    for inst in instances:
        try:
            c_url = f"{inst}/api/v1/captions/{video_id}"
            c_req = urllib.request.Request(c_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(c_req, timeout=8) as c_resp:
                c_data = json.loads(c_resp.read().decode("utf-8"))
            
            captions = c_data.get("captions", [])
            if captions:
                target_c = captions[0]
                sub_download_url = f"{inst}{target_c.get('url')}"
                vtt_req = urllib.request.Request(sub_download_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(vtt_req, timeout=8) as vtt_resp:
                    vtt_text = vtt_resp.read().decode("utf-8", errors="ignore")
                
                lines = []
                for l in vtt_text.splitlines():
                    l = l.strip()
                    if not l or l.startswith("WEBVTT") or l.startswith("NOTE") or "-->" in l:
                        continue
                    clean_l = html.unescape(re.sub(r"<[^>]+>", "", l)).strip()
                    if clean_l and not (clean_l.startswith("[") and clean_l.endswith("]")):
                        if not lines or lines[-1] != clean_l:
                            lines.append(clean_l)
                
                if lines:
                    lang_label = f"{target_c.get('label', 'Phụ đề')} [{target_c.get('language_code', '')}] • Mạng lưới Proxy"
                    return "\n".join(lines), lang_label, None
        except Exception:
            continue

    return None, None, "Không thể lấy phụ đề do YouTube tạm thời hạn chế truy cập đối với video này. Hãy dán trực tiếp kịch bản vào khung bên dưới."

def get_subtitles(url):
    video_id = extract_video_id(url)
    if not video_id:
        return None, None, "Đường link không hợp lệ hoặc không tìm thấy ID video YouTube."
    return fetch_youtube_subtitles_cloud(video_id)

# --- THANH BÊN (LỊCH SỬ PHIÊN) ---
with st.sidebar:
    st.markdown("### 🗂️ Danh Sách Dự Án")
    c_new, c_clear = st.columns([1.5, 1])
    with c_new:
        if st.button("➕ Tạo mới", use_container_width=True, type="primary"):
            st.session_state.active_id = None
            st.session_state.input_source_text = ""
            st.session_state.detected_sub_lang = ""
            st.rerun()
    with c_clear:
        if st.button("🧹 Xóa hết", use_container_width=True):
            st.session_state.history = []
            st.session_state.active_id = None
            st.session_state.input_source_text = ""
            st.session_state.detected_sub_lang = ""
            st.rerun()
    
    st.divider()
    if not st.session_state.history:
        st.caption("Chưa có dự án nào được lưu.")
    else:
        for item in st.session_state.history:
            is_active = (st.session_state.active_id == item["id"])
            btn_prefix = "👉 🎬" if is_active else "📄"
            btn_label = f"{btn_prefix} {item['title']}"
            if st.button(btn_label, key=f"sidebar_btn_{item['id']}", use_container_width=True):
                st.session_state.active_id = item["id"]
                st.session_state.input_source_text = item.get("source_text", "")
                st.session_state.detected_sub_lang = item.get("orig_lang", "")
                st.rerun()

# --- TIÊU ĐỀ GIAO DIỆN CHÍNH ---
st.markdown("""
<div class="main-header">
    <h1>🎬 OmniContent Studio AI</h1>
    <p>Trích xuất phụ đề gốc YouTube, biên dịch kịch bản & Tối ưu hóa SEO đa nền tảng</p>
</div>
""", unsafe_allow_html=True)

# 1. Chọn phương thức kết nối API
mode = st.selectbox(
    "CHỌN NỀN TẢNG AI ĐIỀU HÀNH:",
    [
        "⚡ 1 API: Groq (Miễn phí 100% - Tốc độ cực cao, Khuyên dùng)",
        "⚡ 1 API: Google Gemini (Bản miễn phí)",
        "⚡ 1 API: Anthropic Claude (Chính hãng hoặc Proxy bên thứ 3)",
        "⚡ 1 API: OpenAI / ChatGPT / Proxy bên thứ 3",
        "🔥 KẾT HỢP 2 API: Gemini Dịch thuật + Groq Tạo Prompt & SEO"
    ],
    key="cfg_mode"
)

# 2. Cấu hình thông tin API
if "Groq" in mode:
    col_key, col_url = st.columns([1.5, 1.2])
    with col_key:
        groq_key = st.text_input("Nhập Groq API Key (gsk_...):", type="password", key="saved_groq_key")
    with col_url:
        st.text_input("Base URL:", value="https://api.groq.com/openai/v1", disabled=True)
    col_model, col_scan = st.columns([2, 1])
    with col_model:
        groq_model = st.selectbox("Chọn Model Groq:", st.session_state.groq_models, key="saved_groq_model")
    with col_scan:
        st.write("")
        st.write("")
        if st.button("🔍 Quét Model Groq", use_container_width=True):
            if not groq_key:
                st.warning("Vui lòng nhập API Key trước khi quét.")
            else:
                try:
                    c = OpenAI(api_key=groq_key.strip(), base_url="https://api.groq.com/openai/v1")
                    m_list = [m.id for m in c.models.list().data if "whisper" not in m.id and "guard" not in m.id and "allam" not in m.id]
                    if m_list:
                        st.session_state.groq_models = m_list
                        st.success("Đã cập nhật danh sách model thành công!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Lỗi: {e}")

elif "Google Gemini" in mode:
    col_key, col_model = st.columns([2, 1])
    with col_key:
        gemini_key = st.text_input("Nhập Google Gemini API Key:", type="password", placeholder="AIzaSy...", key="saved_gemini_key")
    with col_model:
        gemini_model = st.selectbox("Chọn Model:", ["gemini-3.8-flash", "gemini-3-flash-preview"], key="saved_gemini_model")

elif "Claude" in mode:
    col_key, col_url, col_model = st.columns([1.5, 1.2, 1.3])
    with col_key:
        claude_key = st.text_input("Nhập Claude API Key:", type="password", placeholder="sk-ant-api03-...", key="saved_claude_key")
    with col_url:
        claude_url = st.text_input("Base URL bên bán:", value="https://1gw.gwai.cloud/", key="saved_claude_url")
    with col_model:
        claude_model = st.selectbox("Chọn Model:", ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-7-sonnet-20250219"], key="saved_claude_model")

elif "OpenAI" in mode:
    col_key, col_url, col_model = st.columns([1.5, 1.2, 1.3])
    with col_key:
        openai_key = st.text_input("Nhập API Key:", type="password", placeholder="sk-...", key="saved_openai_key")
    with col_url:
        openai_url = st.text_input("Base URL:", placeholder="https://api.openai.com/v1", key="saved_openai_url")
    with col_model:
        openai_model = st.selectbox("Chọn Model:", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], key="saved_openai_model")

else: # KẾT HỢP 2 API
    c1, c2 = st.columns(2)
    with c1:
        gemini_key = st.text_input("Gemini API Key (Dịch kịch bản dài):", type="password", key="saved_combo_gemini")
        gemini_model = "gemini-3.8-flash"
    with c2:
        groq_key = st.text_input("Groq API Key (Tạo 20 Prompt & SEO):", type="password", key="saved_combo_groq")
        groq_model = st.selectbox("Chọn Model Groq:", st.session_state.groq_models, key="saved_combo_groq_model")

# 3. KHU VỰC DÁN LINK YOUTUBE ĐỂ BÓC TÁCH PHỤ ĐỀ
st.markdown("#### 🔗 Trích Xuất Nhanh Phụ Đề Gốc Từ YouTube")
c_yt_link, c_yt_btn = st.columns([3.5, 1])
with c_yt_link:
    yt_input_url = st.text_input("Dán link YouTube:", placeholder="https://www.youtube.com/watch?v=... hoặc link Shorts", label_visibility="collapsed")
with c_yt_btn:
    if st.button("📥 Lấy Phụ Đề", use_container_width=True):
        if not yt_input_url.strip():
            st.warning("Vui lòng dán link video YouTube.")
        else:
            with st.spinner("Đang kích hoạt cổng giải mã phụ đề..."):
                subs_text, lang_name, err_msg = get_subtitles(yt_input_url.strip())
                if err_msg:
                    st.error(err_msg)
                else:
                    st.session_state.input_source_text = subs_text
                    st.session_state.detected_sub_lang = lang_name
                    st.toast(f"Đã lấy phụ đề: {lang_name} 🎉")
                    st.rerun()

if st.session_state.detected_sub_lang:
    st.info(f"🌐 **Ngôn ngữ phụ đề gốc phát hiện được:** `{st.session_state.detected_sub_lang}`")

# 4. KHUNG HIỂN THỊ / CHỈNH SỬA VĂN BẢN NGUỒN
input_text = st.text_area(
    "Nội dung kịch bản nguồn (Tự động điền từ YouTube hoặc tự gõ):", 
    key="input_source_text",
    height=200, 
    placeholder="Nội dung phụ đề sau khi lấy từ video sẽ hiện ở đây, hoặc bạn có thể tự dán kịch bản vào..."
)

# 5. TÙY CHỌN NGÔN NGỮ & ĐỊNH DẠNG
col_lang, col_style, col_btn = st.columns([1.5, 2, 1])
with col_lang:
    target_language = st.selectbox("NGÔN NGỮ ĐÍCH:", LANGUAGES, key="saved_target_lang")
with col_style:
    translation_style = st.selectbox(
        "ĐỊNH DẠNG BẢN DỊCH:",
        [
            "✂️ Chia từng câu ngắn (Khớp nhịp ngắt nghỉ phụ đề gốc)",
            "📖 Văn xuôi liền mạch (Đoạn văn tự nhiên theo tiêu chuẩn đọc truyện)"
        ],
        key="saved_trans_style"
    )
with col_btn:
    st.write("")
    st.write("")
    start_btn = st.button("🚀 BẮT ĐẦU TẠO", use_container_width=True, type="primary")

# --- HÀM BÓC TÁCH KẾT QUẢ METADATA ---
def parse_meta_response(raw_text):
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    data = {"image_prompts": [], "youtube_description": "", "seo_tags": ""}
    
    desc_match = re.search(r"(?:===|###|\*\*)\s*(?:MO_TA_YOUTUBE|MÔ_TẢ_YOUTUBE|MO TA YOUTUBE|MÔ TẢ YOUTUBE|YOUTUBE)[\s:=*]*\n?(.*?)(?=(?:===|###|\*\*)\s*(?:SEO|PROMPT)|$)", text, re.DOTALL | re.IGNORECASE)
    if desc_match:
        data["youtube_description"] = desc_match.group(1).strip()

    seo_match = re.search(r"(?:===|###|\*\*)\s*(?:SEO_TAGS|SEO TAGS|SEO|TAGS|TỪ KHÓA)[\s:=*]*\n?(.*?)(?=(?:===|###|\*\*)\s*(?:PROMPT|MO_TA|MÔ_TẢ)|$)", text, re.DOTALL | re.IGNORECASE)
    if seo_match:
        data["seo_tags"] = seo_match.group(1).strip()

    prompt_match = re.search(r"(?:===|###|\*\*)\s*(?:PROMPT_ANH|PROMPT ANH|PROMPTS|HÌNH ẢNH)[\s:=*]*\n?(.*?)(?=(?:===|###|\*\*)\s*(?:MO_TA|MÔ_TẢ|SEO)|$)", text, re.DOTALL | re.IGNORECASE)
    if prompt_match:
        lines = prompt_match.group(1).strip().split("\n")
        data["image_prompts"] = [re.sub(r"^\d+[\.\)]\s*", "", l).strip() for l in lines if l.strip() and not l.strip().startswith("===")]
    else:
        lines = [line.strip() for line in text.split("\n") if re.match(r"^\d+[\.\)]\s+", line.strip())]
        if lines:
            data["image_prompts"] = [re.sub(r"^\d+[\.\)]\s*", "", l) for l in lines[:20]]
            
    return data

# --- HÀM GỌI DỊCH THUẬT KÈM CƠ CHẾ AUTO-RETRY KHI CHẠM TRẦN 429 ---
def call_single_translation_with_retry(m_choice, chunk_text, lang, style, max_retries=3):
    if "Chia từng câu" in style:
        format_cmd = "Yêu cầu: Tách câu ngắn theo đúng nhịp ngắt nghỉ dòng gốc để làm phụ đề video."
    else:
        format_cmd = "Yêu cầu: Giữ nguyên bố cục đoạn văn xuôi liền mạch, diễn đạt trôi chảy tự nhiên."

    prompt = f"Bạn là một biên dịch viên kịch bản cao cấp. Hãy dịch đoạn văn bản sau sang {lang}.\n{format_cmd}\nChỉ xuất trực tiếp nội dung bản dịch, không viết lời mở đầu hay kết thúc:\n\n{chunk_text}"
    
    for attempt in range(max_retries):
        try:
            if "Groq" in m_choice:
                c = OpenAI(api_key=groq_key.strip(), base_url="https://api.groq.com/openai/v1")
                r = c.chat.completions.create(model=groq_model, messages=[{"role": "user", "content": prompt}], temperature=0.3)
                return r.choices[0].message.content.strip()
            elif "Gemini" in m_choice or "KẾT HỢP" in m_choice:
                genai.configure(api_key=gemini_key.strip())
                m = genai.GenerativeModel(gemini_model)
                return m.generate_content(prompt).text.strip()
            elif "Claude" in m_choice:
                c_url = claude_url.strip().rstrip("/") if claude_url else None
                if c_url and c_url.endswith("/v1"): c_url = c_url[:-3]
                c = anthropic.Anthropic(api_key=claude_key.strip(), base_url=c_url)
                r = c.messages.create(model=claude_model, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
                return r.content[0].text.strip()
            else: # OpenAI
                c = OpenAI(api_key=openai_key.strip(), base_url=openai_url.strip() if openai_url else None)
                r = c.chat.completions.create(model=openai_model, messages=[{"role": "user", "content": prompt}], temperature=0.3)
                return r.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e).lower()
            # Nếu chạm trần Rate Limit (429/TPM), tự động nghỉ rồi gọi lại thay vì sập web
            if ("429" in err_str or "rate limit" in err_str or "quota" in err_str) and attempt < max_retries - 1:
                wait_sec = 6 * (attempt + 1)
                st.toast(f"Đang xả tải hạn mức TPM, tự động thử lại sau {wait_sec}s... ⏳")
                time.sleep(wait_sec)
            else:
                raise e

# --- HÀM DỊCH TỰ ĐỘNG CHIA & GHÉP (CHỈ CHIA 6-7 LẦN KHI GẶP TRUYỆN SIÊU DÀI) ---
def run_translation_with_chunking(m_choice, full_text, lang, style):
    chunks = split_text_smartly(full_text, max_chars=18000)
    total = len(chunks)
    translated_parts = []
    
    for idx, chunk in enumerate(chunks, 1):
        if total > 1:
            st.toast(f"Đang dịch khối {idx}/{total} (Khối lớn ~18.000 ký tự)... ⏳")
        res = call_single_translation_with_retry(m_choice, chunk, lang, style)
        clean_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
        translated_parts.append(clean_res)
        
        # Nghỉ nhẹ 0.8s giữa các khối lớn để giữ nhịp độ an toàn
        if idx < total:
            time.sleep(0.8)
            
    return "\n\n".join(translated_parts)

# --- HÀM TẠO PROMPT & SEO (ĐỒNG BỘ THEO NGÔN NGỮ ĐÍCH) ---
def run_metadata_generation(m_choice, text, lang):
    prompt = f"""
    Dựa vào nội dung kịch bản dưới đây, hãy tạo ĐỦ 3 PHẦN theo đúng cấu trúc thẻ:

    ===PROMPT_ANH===
    (Viết đúng 20 câu Prompt mô tả cảnh chi tiết bằng TIẾNG ANH theo tiến trình câu chuyện để đưa vào AI vẽ ảnh, mỗi prompt 1 dòng, đánh số từ 1 đến 20)

    ===MO_TA_YOUTUBE===
    (Viết tiêu đề video giật tít, tóm tắt diễn biến kịch tính, lời kêu gọi Đăng ký kênh và danh sách hashtag. BẮT BUỘC VIẾT 100% BẰNG {lang})

    ===SEO_TAGS===
    (Danh sách các thẻ từ khóa SEO thịnh hành BẮT BUỘC BẰNG {lang}, cách nhau bằng dấu phẩy)

    KỊCH BẢN:
    {text[:4500]}
    """
    if "Groq" in m_choice or "KẾT HỢP" in m_choice:
        c = OpenAI(api_key=groq_key.strip(), base_url="https://api.groq.com/openai/v1")
        r = c.chat.completions.create(model=groq_model, messages=[{"role": "user", "content": prompt}], temperature=0.3)
        return parse_meta_response(r.choices[0].message.content)
    elif "Gemini" in m_choice:
        genai.configure(api_key=gemini_key.strip())
        m = genai.GenerativeModel(gemini_model)
        return parse_meta_response(m.generate_content(prompt).text)
    elif "Claude" in m_choice:
        c_url = claude_url.strip().rstrip("/") if claude_url else None
        if c_url and c_url.endswith("/v1"): c_url = c_url[:-3]
        c = anthropic.Anthropic(api_key=claude_key.strip(), base_url=c_url)
        r = c.messages.create(model=claude_model, max_tokens=3000, messages=[{"role": "user", "content": prompt}])
        return parse_meta_response(r.content[0].text)
    else: # OpenAI
        c = OpenAI(api_key=openai_key.strip(), base_url=openai_url.strip() if openai_url else None)
        r = c.chat.completions.create(model=openai_model, messages=[{"role": "user", "content": prompt}], temperature=0.3)
        return parse_meta_response(r.choices[0].message.content)

# --- SỰ KIỆN TẠO NỘI DUNG ---
if start_btn:
    current_source = st.session_state.get("input_source_text", "").strip()
    if not current_source:
        st.warning("Vui lòng dán kịch bản hoặc lấy phụ đề từ YouTube.")
    elif "KẾT HỢP" in mode and (not gemini_key or not groq_key):
        st.error("Chế độ kết hợp yêu cầu nhập đủ cả Gemini Key và Groq Key!")
    else:
        try:
            with st.spinner(f"⏳ Bước 1/2: Đang biên dịch kịch bản sang {target_language} (Khối lớn ~90% công suất)..."):
                trans_clean = run_translation_with_chunking(mode, current_source, target_language, translation_style)
            
            with st.spinner(f"⏳ Bước 2/2: Đang tạo 20 Prompt ảnh và SEO/Mô tả bằng {target_language}..."):
                meta_result = run_metadata_generation(mode, current_source, target_language)

            final_data = {
                "translation": trans_clean,
                "image_prompts": meta_result["image_prompts"],
                "youtube_description": meta_result["youtube_description"],
                "seo_tags": meta_result["seo_tags"]
            }

            now_str = datetime.now().strftime("%H:%M - %d/%m")
            unique_id = f"session_{int(time.time() * 1000)}"
            new_session = {
                "id": unique_id,
                "title": f"Dự án {now_str}",
                "lang": target_language,
                "orig_lang": st.session_state.detected_sub_lang,
                "source_text": current_source,
                "data": final_data
            }
            st.session_state.history.insert(0, new_session)
            st.session_state.active_id = unique_id
            st.toast("Tạo nội dung thành công! 🎉")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi: {e}")

# --- HIỂN THỊ KẾT QUẢ PHIÊN ĐANG CHỌN ---
active_session = None
if st.session_state.active_id is not None:
    active_session = next((item for item in st.session_state.history if item["id"] == st.session_state.active_id), None)

if active_session:
    data = active_session["data"]
    s_id = active_session["id"]
    
    st.divider()
    
    # KHUNG ĐỔI TÊN & XÓA PHIÊN
    with st.container():
        orig_info_str = f" • Gốc: {active_session.get('orig_lang')}" if active_session.get('orig_lang') else ""
        st.markdown(f"#### 📌 Đang xem: **{active_session['title']}** — *[{active_session.get('lang', 'Ngôn ngữ')}{orig_info_str}]*")
        c_title_input, c_btn_save, c_btn_del = st.columns([3, 1, 1])
        with c_title_input:
            edit_title = st.text_input(
                "Tên phiên:", 
                value=active_session["title"], 
                key=f"rename_input_{s_id}", 
                label_visibility="collapsed",
                placeholder="Đặt lại tên cho phiên..."
            )
        with c_btn_save:
            if st.button("💾 Lưu tên", key=f"btn_save_{s_id}", use_container_width=True):
                if edit_title.strip():
                    active_session["title"] = edit_title.strip()
                    st.toast(f"Đã cập nhật: {edit_title.strip()}! ✅")
                    st.rerun()
        with c_btn_del:
            if st.button("🗑️ Xóa", key=f"btn_del_{s_id}", use_container_width=True):
                st.session_state.history = [s for s in st.session_state.history if s["id"] != s_id]
                st.session_state.active_id = None
                st.session_state.input_source_text = ""
                st.session_state.detected_sub_lang = ""
                st.rerun()

    # 1. BẢN DỊCH
    with st.expander(f"01. Kịch Bản Đã Dịch ({active_session.get('lang', 'Mục tiêu')})", expanded=True):
        st.text_area("Bản dịch:", value=data.get("translation", ""), height=260, key=f"trans_area_{s_id}")
    
    # 2. 20 PROMPT HÌNH ẢNH
    with st.expander("02. 20 Prompt Hình Ảnh Theo Cảnh (English)", expanded=True):
        prompts = data.get("image_prompts", [])
        if prompts:
            for i, p in enumerate(prompts, 1):
                st.markdown(f"**Cảnh {i:02d}:**")
                st.code(p, language="text")
        else:
            st.warning("Chưa trích xuất được danh sách prompt ảnh.")
            
    # 3. MÔ TẢ YOUTUBE
    with st.expander(f"03. Tiêu Đề & Mô Tả YouTube ({active_session.get('lang', 'Mục tiêu')})", expanded=True):
        st.text_area("Mô tả:", value=data.get("youtube_description", ""), height=160, key=f"desc_area_{s_id}")
        
    # 4. TỪ KHÓA SEO
    with st.expander(f"04. Thẻ Từ Khóa SEO Viral ({active_session.get('lang', 'Mục tiêu')})", expanded=True):
        if data.get("seo_tags"):
            st.info(data.get("seo_tags", ""))
        else:
            st.caption("Chưa có thẻ SEO.")
elif st.session_state.history:
    st.divider()
    st.info("💡 **Bạn đang ở trạng thái Tạo Mới**: Hãy dán link YouTube (bấm Lấy Phụ Đề) hoặc gõ văn bản trực tiếp, sau đó nhấn **BẮT ĐẦU TẠO**.")
