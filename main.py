import streamlit as st
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import pandas as pd
import time

# Загрузка лексикона для VADER (важно для работы с SentimentIntensityAnalyzer)
nltk.download('vader_lexicon')

# Инициализация Reddit API
reddit = praw.Reddit(
    client_id='pF6Ub7vXp1l4MGBjRhrl_g',  # Ваш Client ID
    client_secret='P7RDX_B0qmJZRc2yUD60QJ1VRfbzug',  # Ваш Client Secret
    user_agent='myapp:v1.0 (by u/ergaza1)'  # Ваш user-agent
)

# Инициализация анализатора настроений VADER
sia = SentimentIntensityAnalyzer()


# Функция для предобработки текста (удаляем ненужные символы и ссылки)
def preprocess_text(text):
    text = text.replace("\n", " ").replace("\r", "")  # Убираем переносы строк
    return text


# Функция для получения данных с Reddit
def get_reddit_data(subreddit_name, limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    try:
        # Получаем топовые посты
        for submission in subreddit.top(limit=limit):
            text = preprocess_text(submission.title + " " + submission.selftext)  # Объединяем заголовок и текст
            sentiment = sia.polarity_scores(text)  # Анализ настроений
            posts.append({
                'title': submission.title,
                'text': submission.selftext,
                'sentiment': sentiment['compound'],  # Основной показатель настроения
                'url': submission.url
            })
    except Exception as e:
        st.error(f"Ошибка при получении данных с Reddit: {e}")
        return []

    return posts


# Интерфейс с Streamlit
st.title('Анализ общественного мнения по политическим событиям')

# Выбор сабреддита
subreddit_name = st.text_input('Введите название сабреддита', 'politics')

# Получаем данные с Reddit
limit = st.slider('Количество постов для анализа', 5, 50, 10)
posts = get_reddit_data(subreddit_name, limit)

if posts:  # Если данные успешно получены
    # Преобразуем в DataFrame для удобной работы
    df = pd.DataFrame(posts)

    # Отображаем таблицу с результатами
    st.write(df[['title', 'sentiment', 'url']])

    # Визуализируем изменения настроений
    # Вместо 'title' используем индекс поста для оси X
    df['index'] = df.index  # Добавляем индекс как столбец
    fig = px.line(df, x='index', y='sentiment', title=f'Настроения по постам из {subreddit_name}')
    fig.update_xaxes(tickmode='array', tickvals=df.index, ticktext=df['title'].values)  # Добавляем подписи на оси X
    st.plotly_chart(fig)

    # Добавим фильтрацию по настроению
    st.subheader('Фильтровать по настроению')
    sentiment_filter = st.radio('Выберите тип настроения', ('Все', 'Положительное', 'Негативное', 'Нейтральное'))

    if sentiment_filter == 'Положительное':
        df = df[df['sentiment'] > 0.05]
    elif sentiment_filter == 'Негативное':
        df = df[df['sentiment'] < -0.05]
    elif sentiment_filter == 'Нейтральное':
        df = df[(df['sentiment'] >= -0.05) & (df['sentiment'] <= 0.05)]

    # Отображаем отфильтрованные данные
    st.write(df[['title', 'sentiment', 'url']])

    # Визуализируем график для отфильтрованных данных
    fig_filtered = px.line(df, x='index', y='sentiment',
                           title=f'Настроения по постам из {subreddit_name} (отфильтрованные)')
    fig_filtered.update_xaxes(tickmode='array', tickvals=df.index,
                              ticktext=df['title'].values)  # Добавляем подписи на оси X
    st.plotly_chart(fig_filtered)
else:
    st.write("Нет данных для отображения.")
