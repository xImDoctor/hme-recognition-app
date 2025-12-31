import streamlit as st
from src.ui.sidebar import render_sidebar
from src.ui.tab_recognition import render_recognition_tab



# Импорт дополнительных табов (в т.ч. исключаемых мной для продакшена)
# отлавливаем исключение при ошибке импорта
try:
    from src.ui.tab_about import render_about_tab
    from src.ui.tab_metrics import render_metrics_tab
    HAS_ADDITIONAL_TABS = True
except ImportError:
    HAS_ADDITIONAL_TABS = False



# Page config
st.set_page_config(
    page_title="Распознавание рукописных математических выражений",
    page_icon="🔣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "canvas_result" not in st.session_state:
    st.session_state.canvas_result = None

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

# Sidebar
selected_model = render_sidebar()

if selected_model is None:
    st.error("Не удалось загрузить модель. Проверьте конфигурацию.")
    st.stop()

# Main tabs
if HAS_ADDITIONAL_TABS:
    tab1, tab2, tab3 = st.tabs([
        "Распознавание",
        "Метрики обучения",
        "О приложении"
    ])

    with tab1:
        render_recognition_tab(selected_model)

    with tab2:
        render_metrics_tab()

    with tab3:
        render_about_tab()
else:
    # Если дополнительные табы не импортированы, то только распознавание
    render_recognition_tab(selected_model)
