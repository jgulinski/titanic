# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

sex_d={1:"Mężczyzna", 0:"Kobieta"}
pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="???")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAGR-CWq865Kj_vgex7F-nonwskrz349om5Q&usqp=CAU")

	with overview:
		st.title("Sprawdz czy przeżyłbyś katastrofę Titanica 🚢")

	with left:
		sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		pclass_radio = st.radio( "Klasa", list(pclass_d.keys()), index=2, format_func=lambda x : pclass_d[x] )
		embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )

	with right:
		age_slider = st.slider("Wiek", value=1, min_value=1, max_value=70)
		sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=0, max_value=8)
		parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=0, max_value=7)
		fare_slider = st.slider("Cena biletu", min_value=0, max_value=300, step=1)

	data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby katastrofę?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
