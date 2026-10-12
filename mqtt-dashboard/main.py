import paho.mqtt.client as mqtt
import streamlit as st

st.title("Streamlit Real-Time MQTT Dashboard")

# Configuration via Sidebar
broker_host = st.sidebar.text_input("MQTT Broker", "broker.hivemq.com")
broker_port = st.sidebar.number_input("Port", value=1883)
topic = st.sidebar.text_input("Topic", "home/sensor/temperature")


# Initialize session state for storing latest message
if "mqtt_message" not in st.session_state:
    st.session_state["mqtt_message"] = "Waiting for data..."


# Callback when message arrives
def on_message(client, userdata, msg):
    st.session_state["mqtt_message"] = msg.payload.decode("utf-8")


@st.cache_resource
def get_mqtt_client():
	client = mqtt.Client()
	client.on_message = on_message
	client.connect(broker_host, broker_port, 60)
	client.subscribe(topic)
	client.loop_start()
	return client


client = get_mqtt_client()

st.metric("Latest Payload", st.session_state["mqtt_message"])

# Rerun periodically or use st.empty() to refresh UI view
if st.button("Refresh View"):
	st.rerun()
