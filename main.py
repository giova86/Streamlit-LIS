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

image = Image.open('LIS-MANI.png')


lotties_coding = "https://assets7.lottiefiles.com/packages/lf20_S6vWEd.json"

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

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
words = []

class VideoProcessor:
    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")

        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:

            h, w, c = frame.shape

            # make detection
            image, results = mediapipe_detection(frame, holistic)

            color = (0,0,0)
            #cv2.rectangle(frame, (0+int(0.03*h),int(h-0.14*h)), (0+int(0.75*h), int(h-0.015*h)), color,-1)
            cv2.rectangle(frame, (0, 0),
                                 (int(w*0.18), int(h-h*0.12)), (255,255,255),-1)

            w_i = int(h/len(labels))

            for i in range(len(labels)):
    #            cv2.rectangle(frame, (90, 10+ i*int(50)), (90, 60+ i*int(50)), color,-1)
                cv2.putText(frame, labels[i], (50, (i+1)*int(h/(len(labels)+4))), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (90, (i)*int(h/(len(labels)+4))+30),
                                     (90, (i+1)*int(h/(len(labels)+4)) ), color,-1)

            # perform prediction with relative probability
            if results.right_hand_landmarks:

                # draw_limit_rh(frame, results)

                # uncomment for NN
                # prediction = labels[np.argmax(model.predict(np.array([points_detection(results)])))]

                prediction = model.predict(np.array([points_detection(results)]))[0]
                pred_prob = np.max(model.predict_proba(np.array([points_detection(results)])))

                for i in range(len(labels)):
    #                cv2.rectangle(frame, (70, 10+ i*int(50)), (70+int(model.predict_proba(np.array([points_detection(results)]))[0][i]*100)*3, 60+ i*int(50)), color,-1)
                    cv2.putText(frame, labels[i], (50, (i+1)*int(h/(len(labels)+4))), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (90, (i)*int(h/(len(labels)+4))+30),
                                         (90+int(model.predict_proba(np.array([points_detection(results)]))[0][i]*60)*2, (i+1)*int(h/(len(labels)+4)) ), color,-1)

                # uncomment for NN
                # for i in range(len(labels)):
                #     cv2.rectangle(frame, (70, 10+ i*int(50)), (70+int(model.predict(np.array([points_detection(results)]))[0][i]*100)*3, 60+ i*int(50)), color,-1)
                #     cv2.putText(frame, labels[i], (10, 50+ i*int(50)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 4, cv2.LINE_AA)


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
                            (w-int(0.5*h),int(0.05*h)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0,0,0),
                            2,
                            cv2.LINE_AA)




            return av.VideoFrame.from_ndarray(frame, format='bgr24')


st.write('''
# Alfabeto LIS
##### *Metti alla prova il tuo linguaggio dei segni*
''')

col1, col2, col3 = st.columns((2,5,2))

with col1:
    st.write(' ')

with col2:
    st.image(image)

with col3:
    st.write(' ')



st.write("La LIS non è una forma abbreviata di italiano, una mimica, un qualche codice morse o braille, un semplice alfabeto manuale o un supporto all’espressione della lingua parlata, ma una lingua con proprie regole grammaticali, sintattiche, morfologiche e lessicali. Si è evoluta naturalmente, come tutte le lingue, ma con una struttura molto diversa, che utilizza sia componenti manuali (es. la configurazione, la posizione, il movimento delle mani) che non-manuali, quali l’espressione facciale, la postura, ecc. Ha meccanismi di dinamica evolutiva e di variazione nello spazio (i “dialetti”), e rappresenta un importante strumento di trasmissione culturale. È una lingua che viaggia sul canale visivo-gestuale, integro nelle persone sorde, e ciò consente loro pari opportunità di accesso alla comunicazione.")
st.write("---")

st.header("Test")


col1, col2, col3 = st.columns((1,5,1))

with col1:
    st.write(' ')

with col2:
    webrtc_streamer(key='key', video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": 1080,
                "frameRate": 30
            }
        },)

with col3:
    st.write(' ')


st.write("---")

st.header("GitHub Project")
left_column, right_column = st.columns((1,2))
with right_column:
        st.write("##")
        st.write(
        """
        On my GitHub channel you can find the repository of the LIS project:
        - djskalò djasdjk sa
        - jdkaslò jdsaòlkjd asòkjds aòklds
        - hda slhkjd sahldsa hdsalkj
        """
        )
with left_column:
    lottie_json = load_lottieurl(lotties_coding)
    st_lottie(lottie_json, height=300, key='coding')

st.write("---")

st.header("Bibliography")
left_column, right_column = st.columns((2,1))
with left_column:
        st.write("##")
        st.write(
        """
        On my GitHub channel you can find the repository of the LIS project:
        - djskalò djasdjk sa
        - jdkaslò jdsaòlkjd asòkjds aòklds
        - hda slhkjd sahldsa hdsalkj
        """
        )
with right_column:
    lottie_json = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_1a8dx7zj.json")
    st_lottie(lottie_json, height=200, key='coding2')
#
#
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()
#
# lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
# lottie_json = load_lottieurl(lotties_coding)
# st_lottie(lottie_json)
