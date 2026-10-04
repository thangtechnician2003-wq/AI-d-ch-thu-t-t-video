import streamlit as st
import re
import time
from datetime import datetime

# Import các thư viện AI
import google.generativeai as genai
from openai import OpenAI
import anthropic

# Cấu hình giao diện trang web
st.set_page_config(page_title="ATH - Multi AI Content Gen", layout="wide", page_icon="⚡")

# Khởi tạo bộ nhớ tạm
if "history" not in st.session_state:
    st.session_state.history = []
if "active_idx" not in st.session_state:
    st.session_state.active_idx = None
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

# --- MENU TRÁI (LỊCH SỬ VÀ QUẢN LÝ PHIÊN) ---
with st.sidebar:
    st.title("📁 Lịch Sử Phiên")
    c_new, c_clear = st.columns([1.5, 1])
    with c_new:
        if st.button("➕ Phiên Mới", use_container_width=True, type="primary"):
            st.session_state.active_idx = None
            st.rerun()
    with c_clear:
        if st.button("🧹 Xóa hết", use_container_width=True):
            st.session_state.history = []
            st.session_state.active_idx = None
            st.rerun()
    
    st.divider()
    if not st.session_state.history:
        st.caption("Chưa có phiên làm việc nào.")
    else:
        for idx, item in enumerate(st.session_state.history):
            is_active = (st.session_state.active_idx == idx)
            btn_prefix = "👉 📄" if is_active else "📄"
            btn_label = f"{btn_prefix} {item['title']}"
            if st.button(btn_label, key=f"sidebar_item_{item['id']}", use_container_width=True):
                st.session_state.active_idx = idx
                st.rerun()

# --- GIAO DIỆN CHÍNH ---
st.title("ATH - MULTI AI CONTENT GEN")
st.caption("HỆ THỐNG TỰ ĐỘNG TẠO NỘI DUNG ĐA NỀN TẢNG (1 API HOẶC NHIỀU API)")

# 1. Chọn chế độ cấu hình API
mode = st.selectbox(
    "CHỌN PHƯƠNG THỨC HOẠT ĐỘNG:",
    [
        "⚡ DÙNG 1 API DUY NHẤT: Groq (Miễn phí 100% - Siêu nhanh, Khuyên dùng)",
        "⚡ DÙNG 1 API DUY NHẤT: Google Gemini (Miễn phí 20 lần/ngày)",
        "⚡ DÙNG 1 API DUY NHẤT: Anthropic Claude (Chính hãng hoặc Proxy bên thứ 3)",
        "⚡ DÙNG 1 API DUY NHẤT: OpenAI / ChatGPT / Proxy khác",
        "🔥 KẾT HỢP 2 API: Gemini Dịch + Groq làm Prompt/SEO (Chia tải)"
    ]
)

# 2. Khung cấu hình thông tin API
if "Groq (Miễn phí 100%" in mode:
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
        if st.button("🔍 Quét Model khả dụng trên Groq", use_container_width=True):
            if not groq_key:
                st.warning("Vui lòng dán Groq API Key trước.")
            else:
                try:
                    c = OpenAI(api_key=groq_key.strip(), base_url="https://api.groq.com/openai/v1")
                    m_list = [m.id for m in c.models.list().data if "whisper" not in m.id and "guard" not in m.id and "allam" not in m.id]
                    if m_list:
                        st.session_state.groq_models = m_list
                        st.success("Đã cập nhật danh sách Model!")
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
        gemini_key = st.text_input("Gemini API Key (Dùng để dịch truyện dài):", type="password")
        gemini_model = "gemini-3.8-flash"
    with c2:
        groq_key = st.text_input("Groq API Key (Dùng để tạo Prompt & SEO):", type="password")
        groq_model = st.selectbox("Chọn Model Groq:", st.session_state.groq_models)

# 3. Khung nhập văn bản nguồn
input_text = st.text_area(
    "Nội dung văn bản nguồn:", 
    height=180, 
    placeholder="Dán toàn bộ văn bản hoặc kịch bản truyện dài vào đây..."
)

# 4. Tùy chọn Ngôn ngữ & Phong cách dịch
col_lang, col_style, col_btn = st.columns([1.5, 2, 1])
with col_lang:
    target_language = st.selectbox("NGÔN NGỮ MỤC TIÊU:", LANGUAGES)
with col_style:
    translation_style = st.selectbox(
        "PHONG CÁCH BẢN DỊCH:",
        [
            "✂️ Chia nhỏ từng đoạn (1-2 câu/dòng, tối ưu video & phụ đề)",
            "📖 Văn xuôi liền mạch (Đoạn dài tự nhiên, tối ưu đọc truyện)"
        ]
    )
with col_btn:
    st.write("")
    st.write("")
    start_btn = st.button("🚀 BẮT ĐẦU TẠO", use_container_width=True, type="primary")

# --- HÀM BÓC TÁCH KẾT QUẢ TỪ AI ---
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

# --- HÀM DỊCH THUẬT ---
def run_translation(m_choice, text, lang, style):
    if "Chia nhỏ" in style:
        format_cmd = "Yêu cầu: Chia bản dịch thành từng đoạn rất ngắn (mỗi đoạn 1-2 câu, xuống dòng theo thoại nhân vật) để làm phụ đề video."
    else:
        format_cmd = "Yêu cầu: Giữ nguyên bố cục văn xuôi mạch lạc, liên kết tự nhiên theo tiêu chuẩn xuất bản sách truyện."

    prompt = f"Bạn là dịch giả chuyên nghiệp. Hãy dịch toàn bộ văn bản sau sang {lang}.\n{format_cmd}\nChỉ xuất bản dịch, không giải thích:\n\n{text}"
    
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

# --- HÀM TẠO PROMPT VÀ SEO ---
def run_metadata_generation(m_choice, text):
    prompt = f"""
    Dựa vào diễn biến câu chuyện sau, hãy tạo ĐỦ 3 PHẦN theo đúng định dạng thẻ:

    ===PROMPT_ANH===
    (Viết đúng 20 prompt chi tiết bằng tiếng Anh theo mạch truyện cho AI vẽ ảnh, mỗi prompt 1 dòng, đánh số từ 1 đến 20)

    ===MO_TA_YOUTUBE===
    (Tiêu đề video giật tít, tóm tắt hấp dẫn, kêu gọi Đăng ký và danh sách hashtag)

    ===SEO_TAGS===
    (Danh sách các thẻ từ khóa SEO cách nhau bởi dấu phẩy)

    CÂU CHUYỆN:
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

# --- XỬ LÝ SỰ KIỆN TẠO NỘI DUNG ---
if start_btn:
    if not input_text.strip():
        st.warning("Vui lòng dán nội dung văn bản nguồn.")
    elif "KẾT HỢP" in mode and (not gemini_key or not groq_key):
        st.error("Chế độ kết hợp yêu cầu nhập cả Gemini Key và Groq Key!")
    else:
        try:
            with st.spinner("⏳ Bước 1/2: Đang dịch thuật câu chuyện..."):
                trans_result = run_translation(mode, input_text, target_language, translation_style)
                trans_clean = re.sub(r"<think>.*?</think>", "", trans_result, flags=re.DOTALL).strip()
            
            with st.spinner("⏳ Bước 2/2: Đang tạo 20 Prompt ảnh, Mô tả và Thẻ SEO..."):
                meta_result = run_metadata_generation(mode, input_text)

            final_data = {
                "translation": trans_clean,
                "image_prompts": meta_result["image_prompts"],
                "youtube_description": meta_result["youtube_description"],
                "seo_tags": meta_result["seo_tags"]
            }

            now_str = datetime.now().strftime("%H:%M - %d/%m")
            unique_id = f"{int(time.time()*1000)}"
            new_session = {
                "id": unique_id,
                "title": f"Phiên {now_str}",
                "lang": target_language,
                "data": final_data
            }
            # Thêm phiên mới vào đầu danh sách
            st.session_state.history.insert(0, new_session)
            st.session_state.active_idx = 0
            st.toast("Tạo nội dung thành công! 🎉")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi: {e}")

# --- HIỂN THỊ KẾT QUẢ VÀ QUẢN LÝ PHIÊN ---
if st.session_state.active_idx is not None and st.session_state.history:
    curr_idx = st.session_state.active_idx
    active_session = st.session_state.history[curr_idx]
    data = active_session["data"]
    s_id = active_session["id"]
    
    st.divider()
    
    # KHUNG ĐỔI TÊN VÀ XÓA PHIÊN (ĐÃ SỬA DỨT ĐIỂM BẰNG FORM ĐỘC LẬP)
    with st.container():
        st.write(f"### 📌 Đang xem: **{active_session['title']}**")
        c_title_input, c_btn_save, c_btn_del = st.columns([3, 1, 1])
        with c_title_input:
            edit_title = st.text_input(
                "Đổi tên phiên:", 
                value=active_session["title"], 
                key=f"input_box_{s_id}", 
                label_visibility="collapsed",
                placeholder="Gõ tên mới cho phiên làm việc..."
            )
        with c_btn_save:
            if st.button("💾 Lưu tên", key=f"btn_save_{s_id}", use_container_width=True):
                if edit_title.strip():
                    active_session["title"] = edit_title.strip()
                    st.toast(f"Đã đổi tên thành: {edit_title.strip()}! ✅")
                    st.rerun()
        with c_btn_del:
            if st.button("🗑️ Xóa phiên", key=f"btn_del_{s_id}", use_container_width=True):
                st.session_state.history.pop(curr_idx)
                st.session_state.active_idx = 0 if st.session_state.history else None
                st.rerun()

    # 1. Bản dịch
    with st.expander(f"01. Bản Dịch ({active_session.get('lang', 'Mục tiêu')})", expanded=True):
        st.text_area("Bản dịch:", value=data.get("translation", ""), height=260)
    
    # 2. 20 Prompt ảnh
    with st.expander("02. 20 Prompt Hình Ảnh (English)", expanded=True):
        prompts = data.get("image_prompts", [])
        if prompts:
            for i, p in enumerate(prompts, 1):
                st.markdown(f"**Prompt {i}:**")
                st.code(p, language="text")
        else:
            st.warning("Chưa trích xuất được danh sách prompt ảnh.")
            
    # 3. Mô tả YouTube
    with st.expander("03. Mô Tả YouTube", expanded=True):
        st.text_area("Mô tả:", value=data.get("youtube_description", ""), height=160)
        
    # 4. SEO Tags
    with st.expander("04. SEO Tags Viral", expanded=True):
        if data.get("seo_tags"):
            st.info(data.get("seo_tags", ""))
        else:
            st.caption("Chưa có thẻ SEO.")