from streamlit_webrtc import webrtc_streamer
import av
import cv2
from utils import mediapipe_detection, draw_landmarks, draw_landmarks_custom, draw_limit_rh, draw_limit_lh, check_detection, points_detection
import pickle
from sklearn import svm
import numpy as np
import mediapipe as mp
import time
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image

custom_css = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;
}
.stAppHeader a {
    display: none;
}
.appview-container .main .block-container{
    max-width: 900px;
}
</style>
'''

st.markdown(custom_css, unsafe_allow_html=True)


image = Image.open('LIS-MANI.png')
image_alfabeto = Image.open('alfabeto.jpg')

# lotties_coding = "https://assets7.lottiefiles.com/packages/lf20_S6vWEd.json"
# def load_lottieurl(url):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

#st.set_page_config(layout='wide')

def local_CSS(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_CSS('style.css')

# load svm model
model = pickle.load(open('models/model_svm_all.sav', 'rb'))
labels = np.array(model.classes_) # put the entire alphabet in the future

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5
                                )

# class VideoProcessor:

def recv(frame):
    frame = frame.to_ndarray(format="bgr24")

    h, w, c = frame.shape

    # make detection
    image, results = mediapipe_detection(frame, holistic)

    # perform prediction with relative probability
    if results.right_hand_landmarks:
        if rh_area:
            draw_limit_rh(frame, results)

        prediction = model.predict(np.array([points_detection(results)]))[0]
        pred_prob = np.max(model.predict_proba(np.array([points_detection(results)])))

        # add text with prediction
        if pred_prob > 0.5:
            cv2.putText(frame, f'{prediction.capitalize()} ({int(pred_prob*100)}%)',
                        (0+int(0.05*h),h-int(0.05*h)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2 ,
                        (0,255,0),
                        2,
                        cv2.LINE_AA)
        elif pred_prob < 0.3:
            cv2.putText(frame, 'I am not sure...',
                        (0+int(0.05*h),h-int(0.05*h)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2 ,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA)
        else:
            cv2.putText(frame, f'Maybe {prediction.capitalize()} ({int(pred_prob*100)}%)',
                        (0+int(0.05*h),h-int(0.05*h)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2 ,
                        (45, 255, 255),
                        2,
                        cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Detecting Hand...',
                    (0+int(0.05*h),h-int(0.05*h)),
                    #(w-int(0.5*h),int(0.05*h)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),
                    2,
                    cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(frame, format='bgr24')


st.write('''
# Alfabeto LIS
##### *Metti alla prova il tuo linguaggio dei segni*
''')

col1, col2, col3 = st.columns((2,4,2))

with col1:
    st.write(' ')

with col2:
    st.image(image)

with col3:
    st.write(' ')


st.write("La LIS non è una forma abbreviata di italiano, una mimica, un qualche codice morse o braille, un semplice alfabeto manuale o un supporto all’espressione della lingua parlata, ma una lingua con proprie regole grammaticali, sintattiche, morfologiche e lessicali. Si è evoluta naturalmente, come tutte le lingue, ma con una struttura molto diversa, che utilizza sia componenti manuali (es. la configurazione, la posizione, il movimento delle mani) che non-manuali, quali l’espressione facciale, la postura, ecc. Ha meccanismi di dinamica evolutiva e di variazione nello spazio (i “dialetti”), e rappresenta un importante strumento di trasmissione culturale. È una lingua che viaggia sul canale visivo-gestuale, integro nelle persone sorde, e ciò consente loro pari opportunità di accesso alla comunicazione.")
st.write("---")

st.header("Video Tutorial")
st.write('')
st.write('')
col1, col2, col3, col4 = st.columns([2,9,6,2], gap='Large')
with col2:
    st.video('https://www.youtube.com/watch?v=0Yx9IkOxFyI')
with col3:
    st.image(image_alfabeto)


st.write("---")
st.header("Test")
st.write("Mettiti alla prova con la nostra intelligenza artificiale. La nostra Rete Neurale è ingrado di riconoscere la posizione della tua mano destra ed indicare con qualche probabilità la lettera eseguita vie riconosciuta.")


col1, col2, col3 = st.columns((1,7,1))
with col2:
    st.write('')
    webrtc_streamer(key='key', #video_frame_callback=recv,
        media_stream_constraints={
            "video": {
                "width": 1080,
                "frameRate": 30
            }
        }
    )
    rh_area = st.checkbox('Mostra area mano destra')
