import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- 1. Lógica del Chatbot (Backend) ---
# Esta parte es casi idéntica a la que ya tenías.

# Dataset de conocimientos con fuentes
data = [
    {"pregunta": "primer presidente del peru", "respuesta": "El primer presidente del Perú fue José de la Riva-Agüero en 1823.", "fuente": "https://es.wikipedia.org/wiki/Presidente_del_Per%C3%BA"},
    {"pregunta": "independencia del peru", "respuesta": "La independencia del Perú fue proclamada por José de San Martín el 28 de julio de 1821.", "fuente": "https://es.wikipedia.org/wiki/Independencia_del_Per%C3%BA"},
    {"pregunta": "culturas preincas", "respuesta": "Algunas culturas preincas notables fueron Chavín, Moche, Nazca, Paracas, Tiahuanaco y Wari.", "fuente": "https://es.wikipedia.org/wiki/Culturas_preincaicas"},
    {"pregunta": "batalla de ayacucho", "respuesta": "La Batalla de Ayacucho, el 9 de diciembre de 1824, fue decisiva para sellar la independencia del Perú y de América del Sur.", "fuente": "https://es.wikipedia.org/wiki/Batalla_de_Ayacucho"},
    {"pregunta": "quien fue tupac amaru ii", "respuesta": "Túpac Amaru II (José Gabriel Condorcanqui) lideró la mayor rebelión anticolonial en hispanoamérica en el siglo XVIII.", "fuente": "https://es.wikipedia.org/wiki/T%C3%BApac_Amaru_II"},
    {"pregunta": "el imperio incaico", "respuesta": "El Imperio Incaico o Tahuantinsuyo fue el imperio más grande de la América precolombina, con capital en Cusco.", "fuente": "https://es.wikipedia.org/wiki/Imperio_incaico"},
    {"pregunta": "guerra del pacifico", "respuesta": "La Guerra del Pacífico (1879-1884) fue un conflicto armado que enfrentó a Chile contra una alianza de Perú y Bolivia.", "fuente": "https://es.wikipedia.org/wiki/Guerra_del_Pac%C3%ADfico"},
    {"pregunta": "heroes de la guerra del pacifico", "respuesta": "Algunos héroes peruanos de la Guerra del Pacífico son Miguel Grau, Francisco Bolognesi y Andrés Avelino Cáceres.", "fuente": "https://historiaperuana.pe/periodo-independiente/republica/heroes-guerra-pacifico"},
    {"pregunta": "cuando goberno fujimori", "respuesta": "Alberto Fujimori fue presidente del Perú durante la década de 1990, desde 1990 hasta el año 2000.", "fuente": "https://es.wikipedia.org/wiki/Alberto_Fujimori"},
    {"pregunta": "captura de atahualpa", "respuesta": "Atahualpa, el último emperador inca, fue capturado por los conquistadores españoles liderados por Francisco Pizarro en Cajamarca en 1532.", "fuente": "https://es.wikipedia.org/wiki/Captura_de_Atahualpa"},
    {"pregunta": "fundacion de lima", "respuesta": "Lima, la 'Ciudad de los Reyes', fue fundada por Francisco Pizarro el 18 de enero de 1535.", "fuente": "https://es.wikipedia.org/wiki/Historia_de_Lima"},
    {"pregunta": "el virreinato del peru", "respuesta": "El Virreinato del Perú fue una entidad territorial del Imperio español creada en 1542, con capital en Lima.", "fuente": "https://es.wikipedia.org/wiki/Virreinato_del_Per%C3%BA"},
    {"pregunta": "presidente ramon castilla", "respuesta": "Ramón Castilla fue un presidente clave del siglo XIX, conocido por abolir la esclavitud y el tributo indígena en Perú.", "fuente": "https://es.wikipedia.org/wiki/Ram%C3%B3n_Castilla"},
    {"pregunta": "corriente libertadora del sur", "respuesta": "La Corriente Libertadora del Sur, liderada por José de San Martín, fue clave para la independencia de Argentina, Chile y Perú.", "fuente": "https://es.wikipedia.org/wiki/Corriente_Libertadora_del_Sur"},
    {"pregunta": "corriente libertadora del norte", "respuesta": "La Corriente Libertadora del Norte, liderada por Simón Bolívar, consolidó la independencia de Venezuela, Colombia, Ecuador, Perú y Bolivia.", "fuente": "https://es.wikipedia.org/wiki/Corriente_Libertadora_del_Norte"},
    {"pregunta": "que fue la cultura chavin", "respuesta": "La cultura Chavín fue una civilización del Antiguo Perú que se desarrolló durante el Horizonte Temprano (1200 a. C.-400 a. C.). Su centro ceremonial fue Chavín de Huántar.", "fuente": "https://es.wikipedia.org/wiki/Cultura_chav%C3%ADn"},
    {"pregunta": "que son las lineas de nazca", "respuesta": "Las líneas de Nazca son antiguos geoglifos ubicados en el desierto de Nazca, creados por la cultura Nazca. Representan figuras de animales, plantas y formas geométricas.", "fuente": "https://es.wikipedia.org/wiki/L%C3%ADneas_de_Nazca"},
    {"pregunta": "quien fue pachacutec", "respuesta": "Pachacútec fue el noveno gobernante del estado Inca y quien lo convirtió de un simple curacazgo a un gran imperio: el Tahuantinsuyo.", "fuente": "https://es.wikipedia.org/wiki/Pachac%C3%BAtec"},
    {"pregunta": "que fue el senor de sipan", "respuesta": "El Señor de Sipán fue un antiguo gobernante mochica del siglo III. Su tumba, encontrada intacta, es uno de los descubrimientos arqueológicos más importantes del Perú.", "fuente": "https://es.wikipedia.org/wiki/Se%C3%B1or_de_Sip%C3%A1n"},
    {"pregunta": "combate de angamos", "respuesta": "El Combate de Angamos ocurrió el 8 de octubre de 1879, donde el monitor Huáscar, comandado por Miguel Grau, se enfrentó a la escuadra chilena. Grau murió en este combate.", "fuente": "https://es.wikipedia.org/wiki/Combate_de_Angamos"},
    {"pregunta": "batalla de arica", "respuesta": "En la Batalla de Arica, el 7 de junio de 1880, las tropas peruanas bajo el mando de Francisco Bolognesi defendieron el morro de Arica. Es famosa por la frase 'Tengo deberes sagrados que cumplir y los cumpliré hasta quemar el último cartucho'.", "fuente": "https://es.wikipedia.org/wiki/Batalla_de_Arica"},
    {"pregunta": "que fue la era del guano", "respuesta": "La Era del Guano fue un período de la historia republicana del Perú (1845-1866) donde la exportación de guano de las islas generó una gran bonanza económica.", "fuente": "https://es.wikipedia.org/wiki/Era_del_Guano"},
    {"pregunta": "contrato grace", "respuesta": "El Contrato Grace, firmado en 1889, fue un acuerdo entre el gobierno peruano y acreedores británicos para aliviar la deuda externa a cambio del control de los ferrocarriles por 66 años.", "fuente": "https://es.wikipedia.org/wiki/Contrato_Grace"},
    {"pregunta": "gobierno de augusto b leguia", "respuesta": "El Oncenio de Leguía fue un período de 11 años (1919-1930) en el que Augusto B. Leguía gobernó de forma autoritaria, promoviendo la modernización y la inversión extranjera.", "fuente": "https://es.wikipedia.org/wiki/Oncenio_de_Legu%C3%ADa"},
    {"pregunta": "conflicto del falso paquisha", "respuesta": "El conflicto del Falso Paquisha fue un enfrentamiento armado entre Perú y Ecuador en 1981 en la Cordillera del Cóndor, por un puesto de vigilancia que Ecuador había establecido en territorio peruano.", "fuente": "https://es.wikipedia.org/wiki/Conflicto_del_Falso_Paquisha"},
    {"pregunta": "guerra del cenepa", "respuesta": "La Guerra del Cenepa fue el último conflicto armado entre Perú y Ecuador, ocurrido a principios de 1995 en la zona del río Cenepa.", "fuente": "https://es.wikipedia.org/wiki/Guerra_del_Cenepa"},
    {"pregunta": "quien fue javier perez de cuellar", "respuesta": "Javier Pérez de Cuéllar fue un diplomático peruano que se desempeñó como el quinto Secretario General de las Naciones Unidas de 1982 a 1991.", "fuente": "https://es.wikipedia.org/wiki/Javier_P%C3%A9rez_de_Cu%C3%A9llar"},
    {"pregunta": "acuerdo de paz entre peru y ecuador", "respuesta": "El Acta de Brasilia fue el acuerdo de paz firmado en 1998 por los presidentes Alberto Fujimori (Perú) y Jamil Mahuad (Ecuador), que puso fin a la disputa territorial entre ambos países.", "fuente": "https://es.wikipedia.org/wiki/Acta_de_Brasilia"},
    {"pregunta": "que es el terrorismo en el peru", "respuesta": "Fue un período de conflicto armado interno en Perú entre 1980 y 2000, protagonizado principalmente por los grupos terroristas Sendero Luminoso y el MRTA contra el Estado peruano.", "fuente": "https://es.wikipedia.org/wiki/%C3%89poca_del_terrorismo_en_el_Per%C3%BA"},
    {"pregunta": "captura de abimael guzman", "respuesta": "Abimael Guzmán, líder del grupo terrorista Sendero Luminoso, fue capturado en Lima el 12 de septiembre de 1992, en una operación policial conocida como 'Operación Victoria'.", "fuente": "https://es.wikipedia.org/wiki/Abimael_Guzm%C3%A1n"},
    {"pregunta": "que es machu picchu", "respuesta": "Machu Picchu es una ciudadela inca del siglo XV, ubicada en la Cordillera Oriental de los Andes en Perú. Es uno de los sitios arqueológicos más famosos del mundo.", "fuente": "https://es.wikipedia.org/wiki/Machu_Picchu"},
    {"pregunta": "que es el quipu", "respuesta": "El quipu fue un sistema de registro y contabilidad utilizado por los incas, que consistía en cuerdas de lana o algodón con nudos de varios colores y formas.", "fuente": "https://es.wikipedia.org/wiki/Quipu"},
    {"pregunta": "quien fue el virrey toledo", "respuesta": "Francisco de Toledo fue el quinto virrey del Perú (1569-1581) y es considerado el gran organizador del virreinato por las reformas que implementó.", "fuente": "https://es.wikipedia.org/wiki/Francisco_de_Toledo"},
    {"pregunta": "la rebelion de manco inca", "respuesta": "Manco Inca lideró una gran rebelión contra los conquistadores españoles a partir de 1536, incluyendo el asedio de Cusco, en un intento por restaurar el Imperio Inca.", "fuente": "https://es.wikipedia.org/wiki/Manco_Inca"},
    {"pregunta": "quien fue andres avelino caceres", "respuesta": "Andrés Avelino Cáceres, conocido como el 'Brujo de los Andes', fue un héroe de la Guerra del Pacífico que lideró la resistencia en la sierra (Campaña de la Breña) y fue tres veces presidente del Perú.", "fuente": "https://es.wikipedia.org/wiki/Andr%C3%A9s_Avelino_C%C3%A1ceres"},
    {"pregunta": "que fue la republica aristocratica", "respuesta": "La República Aristocrática (1895-1919) fue un período de la historia peruana dominado por una oligarquía dedicada a la agroexportación, minería y finanzas, con exclusión de las clases populares.", "fuente": "https://es.wikipedia.org/wiki/Rep%C3%BAblica_Aristocr%C3%A1tica"},
    {"pregunta": "gobierno de juan velasco alvarado", "respuesta": "Juan Velasco Alvarado lideró un gobierno militar de izquierda (1968-1975) que implementó la Reforma Agraria, nacionalizó empresas estratégicas y promovió cambios sociales profundos.", "fuente": "https://es.wikipedia.org/wiki/Gobierno_Revolucionario_de_la_Fuerza_Armada"},
    {"pregunta": "que fue la reforma agraria", "respuesta": "La Reforma Agraria de 1969, bajo el gobierno de Velasco Alvarado, fue un proceso de expropiación de latifundios y haciendas para redistribuir la tierra a los campesinos y cooperativas.", "fuente": "https://es.wikipedia.org/wiki/Reforma_agraria_peruana_de_1969"},
    {"pregunta": "primer inca", "respuesta": "Según la leyenda, el primer gobernante y fundador del Curacazgo del Cuzco fue Manco Cápac.", "fuente": "https://es.wikipedia.org/wiki/Manco_C%C3%A1pac"},
    {"pregunta": "que es el inti raymi", "respuesta": "El Inti Raymi o 'Fiesta del Sol' era una ceremonia religiosa incaica en honor al dios sol Inti. Hoy en día es una representación teatral que se celebra en Cusco cada 24 de junio.", "fuente": "https://es.wikipedia.org/wiki/Inti_Raymi"},
    {"pregunta": "ultimo inca de vilcabamba", "respuesta": "El último inca de la resistencia de Vilcabamba fue Túpac Amaru I, quien fue capturado y ejecutado por orden del virrey Francisco de Toledo en 1572.", "fuente": "https://es.wikipedia.org/wiki/T%C3%BApac_Amaru_I"},
    {"pregunta": "quien fue jose carlos mariategui", "respuesta": "José Carlos Mariátegui fue un escritor, periodista y pensador político peruano, considerado uno de los más importantes teóricos del marxismo en América Latina. Su obra más famosa es '7 ensayos de interpretación de la realidad peruana'.", "fuente": "https://es.wikipedia.org/wiki/Jos%C3%A9_Carlos_Mari%C3%A1tegui"},
    {"pregunta": "quien fue victor raul haya de la torre", "respuesta": "Víctor Raúl Haya de la Torre fue un pensador y político peruano, fundador de la Alianza Popular Revolucionaria Americana (APRA) y una de las figuras políticas más influyentes del siglo XX en Perú.", "fuente": "https://es.wikipedia.org/wiki/V%C3%ADctor_Ra%C3%BAl_Haya_de_la_Torre"},
    {"pregunta": "conspiracion de los 13 de la fama", "respuesta": "Se refiere al episodio en la Isla del Gallo en 1526, donde Francisco Pizarro trazó una línea en la arena, invitando a sus hombres a cruzarla para continuar la conquista del Perú. Solo trece lo siguieron.", "fuente": "https://es.wikipedia.org/wiki/Trece_de_la_Fama"},
    {"pregunta": "la cultura wari", "respuesta": "La cultura Wari fue una civilización andina que floreció en el centro de los Andes aproximadamente desde el siglo VII hasta el XIII d. C., llegando a expandirse hasta los actuales departamentos peruanos de Lambayeque por el norte, Arequipa por el sur y hasta la selva del departamento del Cuzco por el este.", "fuente": "https://es.wikipedia.org/wiki/Cultura_wari"},
    {"pregunta": "que es caral", "respuesta": "La civilización Caral fue la más antigua de América, desarrollándose entre el 3000 y 1800 a. C. Su centro principal fue la ciudad sagrada de Caral, ubicada en el valle de Supe.", "fuente": "https://es.wikipedia.org/wiki/Civilizaci%C3%B3n_caral"},
    {"pregunta": "el motin de aranjuez", "respuesta": "El Motín de Aranjuez fue un levantamiento popular en 1808 en España que obligó al rey Carlos IV a abdicar en favor de su hijo Fernando VII. Este evento debilitó a la corona española y fue un catalizador para los movimientos de independencia en América.", "fuente": "https://es.wikipedia.org/wiki/Mot%C3%ADn_de_Aranjuez"},
    {"pregunta": "la rebelion de juan santos atahualpa", "respuesta": "Juan Santos Atahualpa lideró una importante rebelión en la selva central del Perú en 1742 contra el dominio español, logrando mantener una zona liberada del control virreinal durante más de una década.", "fuente": "https://es.wikipedia.org/wiki/Juan_Santos_Atahualpa"},
    {"pregunta": "que fue la confederacion peru-boliviana", "respuesta": "Fue un estado constituido por la unión de tres estados (Nor-Peruano, Sud-Peruano y Bolivia) bajo el mando del mariscal Andrés de Santa Cruz. Duró de 1836 a 1839.", "fuente": "https://es.wikipedia.org/wiki/Confederaci%C3%B3n_Per%C3%BA-Boliviana"},
    {"pregunta": "presidente fernando belaunde terry", "respuesta": "Fernando Belaúnde Terry fue un arquitecto y político que fue presidente del Perú en dos mandatos (1963-1968 y 1980-1985). Su segundo gobierno marcó el retorno a la democracia después de 12 años de gobierno militar.", "fuente": "https://es.wikipedia.org/wiki/Fernando_Bela%C3%BAnde_Terry"}
]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

preguntas = [preprocess_text(item["pregunta"]) for item in data]
respuestas_data = {preprocess_text(item["pregunta"]): {"respuesta": item["respuesta"], "fuente": item["fuente"]} for item in data}

vectorizador = TfidfVectorizer()
X = vectorizador.fit_transform(preguntas)

def get_bot_response(pregunta_usuario):
    processed_input = preprocess_text(pregunta_usuario)
    vector_usuario = vectorizador.transform([processed_input])
    similitudes = cosine_similarity(vector_usuario, X)
    idx = similitudes.argmax()
    
    if similitudes[0, idx] > 0.05:
        pregunta_original = preguntas[idx]
        return respuestas_data[pregunta_original]
    else:
        return {
            "respuesta": "Lo siento, solo puedo responder preguntas de Historia del Perú.",
            "fuente": ""
        }

# --- 2. Interfaz de Usuario con Streamlit (Frontend) ---

st.title("Chatbot de Historia del Perú 🇵🇪")
st.caption("Una aplicación para responder preguntas sobre la historia peruana.")

# Inicializar el historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "¡Hola! ¿Qué te gustaría saber sobre la historia del Perú?"})

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "fuente" in message and message["fuente"]:
            st.markdown(f"[Fuente]({message['fuente']})")

# Input del usuario
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # Agregar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar y mostrar respuesta del bot
    with st.chat_message("assistant"):
        response_data = get_bot_response(prompt)
        st.markdown(response_data["respuesta"])
        if response_data["fuente"]:
            st.markdown(f"[Fuente]({response_data['fuente']})")
        
        # Guardar respuesta del bot en el historial
        full_response = {"role": "assistant", "content": response_data["respuesta"], "fuente": response_data["fuente"]}

        st.session_state.messages.append(full_response)

