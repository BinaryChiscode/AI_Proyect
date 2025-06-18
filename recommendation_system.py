#sistema de recomendación de animes
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def recommend_anime(anime_title, anime_data, top_n=10):
    # Verificar si el anime existe en los datos
    if anime_title not in anime_data['title'].values:
        return f"El anime '{anime_title}' no se encuentra en la base de datos."

    # Crear una matriz TF-IDF para las descripciones de los animes
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(anime_data['description'])

    # Calcular la similitud del coseno entre los animes
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Obtener el índice del anime solicitado
    idx = anime_data.index[anime_data['title'] == anime_title].tolist()[0]

    # Obtener las puntuaciones de similitud para todos los animes
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar los animes por puntuación de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de los animes más similares
    sim_indices = [i[0] for i in sim_scores[1:top_n + 1]]

    # Devolver los títulos de los animes recomendados
    return anime_data['title'].iloc[sim_indices].tolist()
# Ejemplo de uso
if __name__ == "__main__":
    # Cargar los datos de animes
    anime_data = pd.read_csv('anime.csv')  # Asegúrate de que el archivo anime.csv esté en el mismo directorio

    # Anime para recomendar
    anime_title = 'Naruto'

    # Obtener recomendaciones
    recommendations = recommend_anime(anime_title, anime_data)

    # Imprimir las recomendaciones
    print(f"Recomendaciones para '{anime_title}':")
    for i, title in enumerate(recommendations, start=1):
        print(f"{i}. {title}")
# Asegúrate de que el archivo anime.csv tenga las columnas 'title' y 'description'
# Puedes descargar un dataset de animes desde Kaggle o cualquier otra fuente confiable
