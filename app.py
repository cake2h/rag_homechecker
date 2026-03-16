import streamlit as st
import os
from core_logic import load_and_process_docs, create_vector_store, get_qa_chain

st.set_page_config(page_title="AI Ассистент для проверки ДЗ", layout="wide")

st.title("🎓 RAG Система проверки домашних работ")

# Сайдбар для управления базой знаний
with st.sidebar:
    st.header("База знаний преподавателя")
    uploaded_files = st.file_uploader(
        "Загрузите методички/критерии (PDF, TXT)", 
        accept_multiple_files=True,
        type=['pdf', 'txt']
    )
    if st.button("Обновить базу знаний"):
        if uploaded_files:
            # Сохраняем временно файлы
            if not os.path.exists('temp_docs'):
                os.makedirs('temp_docs')
            for f in uploaded_files:
                with open(os.path.join('temp_docs', f.name), 'wb') as out:
                    out.write(f.getbuffer())
            
            with st.spinner("Обработка документов..."):
                splits = load_and_process_docs('temp_docs')
                vectorstore = create_vector_store(splits)
                st.session_state['vectorstore'] = vectorstore
            st.success("База знаний обновлена!")
        else:
            st.warning("Загрузите файлы.")

# Основная часть
if 'vectorstore' not in st.session_state:
    st.info("Пожалуйста, сначала загрузите методички в боковом меню.")
else:
    st.subheader("Проверка работы студента")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        student_work = st.text_area(
            "Вставьте текст домашней работы или код:", 
            height=300,
            placeholder="Здесь студент описывает решение задачи или вставляет код..."
        )
    
    with col2:
        task_context = st.text_input(
            "Тема задания или сам вопрос (опционально):",
            placeholder="Например: Лабораторная работа №2, циклы в Python"
        )

    if st.button("Проверить работу", type="primary"):
        if student_work:
            vectorstore = st.session_state['vectorstore']
            qa_chain = get_qa_chain(vectorstore)
            
            # Формируем запрос
            full_query = f"Контекст задания: {task_context}\n\nТекст работы студента:\n{student_work}"
            
            with st.spinner("Анализирую работу..."):
                result = qa_chain.invoke({"query": full_query})
            
            st.markdown("### Результат проверки:")
            st.markdown(result["result"])
            
            with st.expander("📚 На какие источники опирался ИИ?"):
                for doc in result["source_documents"]:
                    st.caption(f"{doc.metadata.get('source', 'Неизвестный источник')}")
                    st.text(doc.page_content[:200] + "...")
        else:
            st.error("Введите текст работы.")