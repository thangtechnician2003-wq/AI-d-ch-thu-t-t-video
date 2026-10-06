import streamlit as st
import re
import time
from datetime import datetime

# Import các thư viện AI
import google.generativeai as genai
from openai import OpenAI
import anthropic

# 1. CẤU HÌNH GIAO DIỆN & STYLE MỚI (LOẠI BỎ HOÀN TOÀN CHỮ ATH)
st.set_page_config(
    page_title="Studio Kịch Bản & Video AI", 
    layout="wide", 
    page_icon="🎬"
)

# Tùy biến giao diện hiện đại hơn
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

# --- THANH BÊN: QUẢN LÝ LỊCH SỬ PHIÊN DỰ ÁN ---
with st.sidebar:
    st.markdown("### 🗂️ Danh Sách Dự Án")
    c_new, c_clear = st.columns([1.5, 1])
    with c_new:
        if st.button("➕ Tạo mới", use_container_width=True, type="primary"):
            st.session_state.active_id = None
            st.rerun()
    with c_clear:
        if st.button("🧹 Xóa hết", use_container_width=True):
            st.session_state.history = []
            st.session_state.active_id = None
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
                st.rerun()

# --- KHUNG BANNER TIÊU ĐỀ MỚI ---
st.markdown("""
<div class="main-header">
    <h1>🎬 OmniContent Studio AI</h1>
    <p>Hệ thống tự động biên dịch kịch bản đa ngữ, tạo 20 Prompt ảnh nghệ thuật & Tối ưu hóa SEO YouTube</p>
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
    ]
)

# 2. Cấu hình thông tin API
if "Groq" in mode:
    col_key, col_url = st.columns([1.5, 1.2])
    with col_key:
        groq_key = st.text_input("Nhập Groq API Key (gsk_...):", type="password")
    with col_url:
        st.text_input("Base URL:", value="https://api.groq.com/openai/v1", disabled=True)
    col_model, col_scan = st.columns([2, 1])
    with col_model:
        groq_model = st.selectbox("Chọn Model Groq:", st.session_state.groq_models)
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
        gemini_key = st.text_input("Nhập Google Gemini API Key:", type="password", placeholder="AIzaSy...")
    with col_model:
        gemini_model = st.selectbox("Chọn Model:", ["gemini-3.8-flash", "gemini-3-flash-preview"])

elif "Claude" in mode:
    col_key, col_url, col_model = st.columns([1.5, 1.2, 1.3])
    with col_key:
        claude_key = st.text_input("Nhập Claude API Key:", type="password", placeholder="sk-ant-api03-...")
    with col_url:
        claude_url = st.text_input("Base URL bên bán:", value="https://1gw.gwai.cloud/")
    with col_model:
        claude_model = st.selectbox("Chọn Model:", ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-7-sonnet-20250219"])

elif "OpenAI" in mode:
    col_key, col_url, col_model = st.columns([1.5, 1.2, 1.3])
    with col_key:
        openai_key = st.text_input("Nhập API Key:", type="password", placeholder="sk-...")
    with col_url:
        openai_url = st.text_input("Base URL:", placeholder="https://api.openai.com/v1")
    with col_model:
        openai_model = st.selectbox("Chọn Model:", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])

else: # KẾT HỢP 2 API
    c1, c2 = st.columns(2)
    with c1:
        gemini_key = st.text_input("Gemini API Key (Dịch truyện dài):", type="password")
        gemini_model = "gemini-3.8-flash"
    with c2:
        groq_key = st.text_input("Groq API Key (Làm 20 Prompt & SEO):", type="password")
        groq_model = st.selectbox("Chọn Model Groq:", st.session_state.groq_models)

# 3. Khung nhập văn bản nguồn
input_text = st.text_area(
    "Văn bản kịch bản hoặc câu chuyện gốc:", 
    height=180, 
    placeholder="Dán toàn bộ nội dung kịch bản văn xuôi hoặc truyện dài cần chuyển đổi vào đây..."
)

# 4. Tùy chọn Ngôn ngữ mục tiêu & Định dạng
col_lang, col_style, col_btn = st.columns([1.5, 2, 1])
with col_lang:
    target_language = st.selectbox("NGÔN NGỮ ĐÍCH:", LANGUAGES)
with col_style:
    translation_style = st.selectbox(
        "ĐỊNH DẠNG BẢN DỊCH:",
        [
            "✂️ Chia từng câu ngắn (Tối ưu phụ đề & khớp khung hình video)",
            "📖 Văn xuôi liền mạch (Đoạn văn tự nhiên theo tiêu chuẩn đọc truyện)"
        ]
    )
with col_btn:
    st.write("")
    st.write("")
    start_btn = st.button("🚀 BẮT ĐẦU TẠO", use_container_width=True, type="primary")

# --- HÀM BÓC TÁCH DỮ LIỆU ---
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

# --- HÀM THỰC HIỆN DỊCH THUẬT ---
def run_translation(m_choice, text, lang, style):
    if "Chia từng câu" in style:
        format_cmd = "Yêu cầu: Tách câu ngắn, mỗi đoạn chỉ 1-2 câu, xuống dòng theo lời thoại nhân vật để làm phụ đề cho video."
    else:
        format_cmd = "Yêu cầu: Giữ nguyên bố cục đoạn văn xuôi liền mạch, hành văn uyển chuyển và chuẩn xác."

    prompt = f"Bạn là một biên dịch viên kịch bản cao cấp. Hãy dịch văn bản sau sang {lang}.\n{format_cmd}\nChỉ xuất trực tiếp nội dung bản dịch, không viết lời mở đầu hay kết thúc:\n\n{text}"
    
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

# --- HÀM TẠO PROMPT & SEO (BẮT BUỘC ĐỒNG BỘ THEO NGÔN NGỮ ĐANG CHỌN) ---
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
    {text[:3500]}
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

# --- XỬ LÝ NÚT BẮT ĐẦU TẠO ---
if start_btn:
    if not input_text.strip():
        st.warning("Vui lòng dán kịch bản hoặc văn bản nguồn.")
    elif "KẾT HỢP" in mode and (not gemini_key or not groq_key):
        st.error("Chế độ kết hợp yêu cầu nhập đủ cả Gemini Key và Groq Key!")
    else:
        try:
            with st.spinner(f"⏳ Đang biên dịch kịch bản sang {target_language}..."):
                trans_result = run_translation(mode, input_text, target_language, translation_style)
                trans_clean = re.sub(r"<think>.*?</think>", "", trans_result, flags=re.DOTALL).strip()
            
            with st.spinner(f"⏳ Đang tạo 20 Prompt ảnh và SEO/Mô tả bằng {target_language}..."):
                meta_result = run_metadata_generation(mode, input_text, target_language)

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
                "data": final_data
            }
            # Thêm phiên mới vào đầu và gán active_id chính xác
            st.session_state.history.insert(0, new_session)
            st.session_state.active_id = unique_id
            st.toast("Tạo nội dung thành công! 🎉")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi: {e}")

# --- KHUNG HIỂN THỊ DỮ LIỆU CỦA PHIÊN ĐƯỢC CHỌN (ĐỘC LẬP THEO ID) ---
if st.session_state.history:
    active_session = next((item for item in st.session_state.history if item["id"] == st.session_state.active_id), None)
    
    if not active_session:
        active_session = st.session_state.history[0]
        st.session_state.active_id = active_session["id"]

    data = active_session["data"]
    s_id = active_session["id"]
    
    st.divider()
    
    # KHUNG ĐỔI TÊN & XÓA PHIÊN (KHÓA THEO ID ĐỂ CHỐNG XUNG ĐỘT)
    with st.container():
        st.markdown(f"#### 📌 Đang xem: **{active_session['title']}** — *[{active_session.get('lang', 'Ngôn ngữ')}]*")
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
                st.session_state.active_id = st.session_state.history[0]["id"] if st.session_state.history else None
                st.rerun()

    # 1. BẢN DỊCH (Có gắn key={s_id} để chống lỗi đệm giữa các phiên cũ/mới)
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
            
    # 3. MÔ TẢ YOUTUBE (Đồng bộ ngôn ngữ)
    with st.expander(f"03. Tiêu Đề & Mô Tả YouTube ({active_session.get('lang', 'Mục tiêu')})", expanded=True):
        st.text_area("Mô tả:", value=data.get("youtube_description", ""), height=160, key=f"desc_area_{s_id}")
        
    # 4. TỪ KHÓA SEO (Đồng bộ ngôn ngữ)
    with st.expander(f"04. Thẻ Từ Khóa SEO Viral ({active_session.get('lang', 'Mục tiêu')})", expanded=True):
        if data.get("seo_tags"):
            st.info(data.get("seo_tags", ""))
        else:
            st.caption("Chưa có thẻ SEO.")
