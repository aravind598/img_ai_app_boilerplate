import streamlit as st
import numpy as np
import time
import tensorflow as tf
from PIL import Image
from img_classifier import prediction, prepare
import traceback
#from img_classifier import our_image_classifier
# import firebase_bro

# Just making sure we are not bothered by File Encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

global models
#model = tf.keras.models.load_model()

@st.experimental_singleton # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    model = tf.keras.models.load_model("enetd0")
    image_array = prepare(image,expand_dims=True)
    image_pred = prediction(model,image_array)
    return str(image_pred)

def main():
    # Metadata for the web app
    st.set_page_config(
    page_title = "Title of the webpage",
    layout = "centered",
    page_icon= ":shark:",
    initial_sidebar_state = "collapsed",
    )
    choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1", # original 10 classes
     "Model 2 (11 food classes)", # original 10 classes + donuts
     "Model 3 (11 food classes + non-food class)") # 11 classes (same as above) + not_food class
    )
    menu = ['Home', 'About', 'Contact', 'Feedback']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        # Let's set the title of our awesome web app
        st.title('Title of your Awesome App')
        # Now setting up a header text
        st.subheader("By Your Cool Dev Name")
        # Option to upload an image file with jpg,jpeg or png extensions
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        
        # When the user clicks the predict button
        if st.button("Predict"):
        # If the user uploads an image
            if uploaded_file is not None:
                # Opening our image
                image = uploaded_file.read()
                print(type(image))
                #image = Image.open(uploaded_file)
                # # Send our image to database for later analysis
                # firebase_bro.send_img(image)
                # Let's see what we got
                st.image(image,use_column_width=True)
                st.write("")
                try:
                    #with st.spinner("The magic of our AI has started...."):
                        #label = our_image_classifier(image)
                    label=make_prediction(image)
                        #time.sleep(8)
                    st.success("We predict this image to be: "+label)
                    #rating = st.slider("Do you mind rating our service?",1,10)
                except Exception as e:
                    st.error(e)
                    st.error(traceback.format_exc())
                    st.error("We apologize something went wrong 🙇🏽‍♂️")
            else:
                st.error("Can you please upload an image 🙇🏽‍♂️")

    elif choice == "Contact":
        # Let's set the title of our Contact Page
        st.title('Get in touch')
        def display_team(name,path,affiliation="",email=""):
            '''
            Function to display picture,name,affiliation and name of creators
            '''
            team_img = Image.open(path)

            st.image(team_img, width=350, use_column_width=False)
            st.markdown(f"## {name}")
            st.markdown(f"#### {affiliation}")
            st.markdown(f"###### Email {email}")
            st.write("------")

        display_team("Your Awesome Name", "./assets/profile_pic.png","Your Awesome Affliation","hello@youareawesome.com")

    elif choice == "About":
        # Let's set the title of our About page
        st.title('About us')

        # A function to display the company logo
        def display_logo(path):
            company_logo = Image.open(path)
            st.image(company_logo, width=350, use_column_width=False)

        # Add the necessary info
        display_logo("./assets/profile_pic.png")
        st.markdown('## Objective')
        st.markdown("Write your company's objective here.")
        st.markdown('## More about the company.')
        st.markdown("Write more about your country here.")

    elif choice == "Feedback":
        # Let's set the feedback page complete with a form
        st.title("Feel free to share your opinions :smile:")

        first_name = st.text_input('First Name:')
        last_name = st.text_input('Last Name:')
        user_email = st.text_input('Enter Email: ')
        feedback = st.text_area('Feedback')

        # When User clicks the send feedback button
        if st.button('Send Feedback'):
            # # Let's send the data to a Database to store it
            # firebase_bro.send_feedback(first_name, last_name, user_email, feedback)

            # Share a Successful Completion Message
            st.success("Your feedback has been shared!")

if __name__ == "__main__":
    main()
