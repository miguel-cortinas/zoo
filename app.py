import joblib
import streamlit as st

# Función para cargar el modelo
def cargar_modelo():
    return joblib.load('models/rfc_clf.pkl')

# Función para procesar los datos y hacer la predicción
def predecir_animal(modelo, datos):
    # Realiza la predicción
    prediction = modelo.predict([datos])[0]
    return prediction

# Cargar el modelo una vez al inicio de la aplicación
modelo = cargar_modelo()

# Interfaz de usuario
st.title('Predicción de clases de animales')

# Preguntas y sus correspondientes características del modelo
preguntas = {
    '¿tiene ojos?': 'eyes',
    '¿Pone huevos?': 'eggs',
    '¿Es acuático?': 'aquatic',
    '¿Es venenoso?': 'venomous',
    '¿tiene patas?': 'legs',
    '¿tiene aletas?': 'fins',
    '¿Es un depredador?': 'predator',
    '¿Puede volar?': 'airborne',
    '¿Es un animal doméstico?': 'domestic',
    '¿Tiene plumas?': 'feathers',
    '¿Tiene dientes?': 'toothed',
    '¿Respira oxígeno?': 'breathes',
    '¿Es del tamaño de un gato?': 'catsize',
    '¿Tiene pelo?': 'hair',
    '¿Tiene cola?': 'tail',
    '¿Produce leche?': 'milk',
    '¿Tiene columna vertebral?': 'backbone'
}

# Almacena las respuestas del usuario en un diccionario
respuestas = {}
for pregunta, caracteristica in preguntas.items():
    respuesta_usuario = st.selectbox(pregunta, ['Sí', 'No'])
    respuestas[caracteristica] = 1 if respuesta_usuario == 'Sí' else 0

# Botón para realizar la predicción
enviar = st.button('Predecir')

# Si el formulario se envía, realiza la predicción y muestra el resultado
if enviar:
    # Convierte las respuestas del usuario en un formato adecuado para el modelo
    datos_usuario = list(respuestas.values())
    
    # Realiza la predicción utilizando la función predecir_animal
    resultado_prediccion = predecir_animal(modelo, datos_usuario)
    
    # Muestra la predicción al usuario
    st.write('el animal pertenece a la clase:', resultado_prediccion)
