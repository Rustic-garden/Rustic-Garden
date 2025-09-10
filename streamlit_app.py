import streamlit as st
import smtplib
import ssl
import os

# Set a title and description for the Streamlit app.
st.title("Booking & Enquiry Form")
st.markdown("Please fill out the form below to confirm your booking or make an enquiry. "
            "Our team will get in touch with you shortly.")

# --- Form Fields ---
st.subheader("Your Details")
name = st.text_input("Name:", placeholder="Enter your full name")
phone = st.text_input("Phone Number:", placeholder="Enter your phone number")

# Add a button to submit the form.
submit_button = st.button("Confirm Booking")

# --- Email Sending Logic ---

def send_email(user_name, user_phone):
    """
    Sends an email with the user's name and phone number.
    Uses Streamlit's secrets management for security.
    """
    try:
        # Get email credentials and recipient from Streamlit secrets.
        # These values must be configured in your .streamlit/secrets.toml file.
        # Example secrets.toml:
        # [email]
        # sender_email = "your-email@gmail.com"
        # password = "your_app_password"
        # recipient_email = "GroupNomad@outlook.com"
        sender_email = st.secrets["email"]["sender_email"]
        password = st.secrets["email"]["password"]
        recipient_email = st.secrets["email"]["recipient_email"]

        subject = "New Booking Request"
        body = f"""
        A new booking request has been received.

        Name: {user_name}
        Phone Number: {user_phone}
        """

        message = f"Subject: {subject}\n\n{body}"

        # Create a secure SSL context
        context = ssl.create_default_context()

        # Connect to the SMTP server and send the email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, message)
        
        return True
    except KeyError:
        st.error("Email configuration secrets are missing. Please check your `.streamlit/secrets.toml` file.")
        return False
    except Exception as e:
        st.error(f"Failed to send email. An error occurred: {e}")
        return False

# When the user clicks the submit button, process the form.
if submit_button:
    # Validate that both fields are filled.
    if name and phone:
        with st.spinner("Sending your request..."):
            # Call the function to send the email.
            if send_email(name, phone):
                st.success("Your booking has been confirmed! We will contact you shortly.")
            else:
                st.error("There was an issue sending your booking confirmation. Please try again later.")
    else:
        st.warning("Please enter both your name and phone number.")
