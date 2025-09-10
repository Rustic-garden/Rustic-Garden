import streamlit as st
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

# Set a title and description for the Streamlit app.
st.title("Booking & Enquiry Form")
st.markdown("Please fill out the form below to confirm your booking or make an enquiry. "
            "Your details will be recorded in our system.")

# --- Google Sheets Connection ---
@st.cache_resource
def get_gspread_client():
    """
    Connects to Google Sheets using credentials from Streamlit secrets.
    """
    try:
        # Load credentials from the secrets file
        credentials_info = st.secrets["gspread"]["credentials"]
        
        # Define the scopes for accessing Google Sheets and Drive
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        
        # Create credentials object
        creds = Credentials.from_service_account_info(credentials_info, scopes=scopes)
        
        # Authorize the gspread client
        client = gspread.authorize(creds)
        
        # Open the specified spreadsheet
        sheet = client.open_by_url(st.secrets["gspread"]["spreadsheet_url"])
        
        return sheet
    except KeyError as e:
        st.error(f"Missing Google Sheets secret key: {e}. Please configure your `.streamlit/secrets.toml` file.")
        return None
    except Exception as e:
        st.error(f"An error occurred while connecting to Google Sheets: {e}")
        return None

# Get the gspread client
sheet = get_gspread_client()

if sheet:
    # Get the specific worksheet
    try:
        worksheet = sheet.worksheet(st.secrets["gspread"]["worksheet_name"])
    except gspread.WorksheetNotFound:
        st.error(f"Worksheet '{st.secrets['gspread']['worksheet_name']}' not found. Please check the name in your `.toml` file.")
        worksheet = None
else:
    worksheet = None

# --- Form Fields ---
st.subheader("Your Details")
name = st.text_input("Name:", placeholder="Enter your full name")
phone = st.text_input("Phone Number:", placeholder="Enter your phone number")

# Add a button to submit the form.
submit_button = st.button("Confirm Booking")

# --- Booking Submission Logic ---
def add_booking(name, phone):
    """
    Adds a new booking row to the Google Sheet.
    """
    if worksheet:
        try:
            # Append a new row to the worksheet
            worksheet.append_row([name, phone, pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")])
            return True
        except Exception as e:
            st.error(f"Failed to add booking to spreadsheet. An error occurred: {e}")
            return False
    return False

# When the user clicks the submit button, process the form.
if submit_button:
    # Validate that both fields are filled.
    if name and phone:
        with st.spinner("Submitting your request..."):
            # Call the function to add the booking to the spreadsheet.
            if add_booking(name, phone):
                st.success("Your booking has been confirmed! We will contact you shortly.")
            else:
                st.error("There was an issue submitting your booking. Please try again later.")
    else:
        st.warning("Please enter both your name and phone number.")

# --- Display Bookings (Optional) ---
# This section is for you to view the bookings directly in the app.
st.subheader("All Bookings")
if worksheet:
    try:
        # Get all records from the worksheet
        bookings_data = worksheet.get_all_records()
        if bookings_data:
            # Create a dataframe and display it
            df = pd.DataFrame(bookings_data)
            st.dataframe(df)
        else:
            st.info("No bookings found yet.")
    except Exception as e:
        st.error(f"Failed to fetch bookings from spreadsheet. An error occurred: {e}")
