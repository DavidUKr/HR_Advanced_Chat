import base64
import streamlit as st
from streamlit_calendar import calendar  # type: ignore

def show_dashboard():

    st.title("Latest News")
    st.write("This is the Dashboard page with announcements.")

    @st.cache_data
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def set_png_as_page_bg(png_file):
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = f'''
        <style>
        body {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            opacity: 0.9;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

    set_png_as_page_bg('./static/images/image.jpg')

    def image_with_position(png_file, top, left, width, height, hyperlink=None):
        bin_str = get_base64_of_bin_file(png_file)
        image_html = f'''
        <div class="image-container" style="position: relative; margin-top: {top}px; margin-left: {left}px;">
            <img src="data:image/png;base64,{bin_str}" style="width: {width}px; height: {height}px;">
        </div>
        '''
        st.markdown(image_html, unsafe_allow_html=True)
        if hyperlink:
            button_html = f'''
            <div class="button-container" style="position: relative; margin-top: -80px; margin-left: 520px;">
                <a href="{hyperlink}" target="_blank">
                    <button style="background-color: #ED82E6; color: white; border: none; padding: 10px 20px; cursor: pointer;">View Document</button>
                </a>
            </div>
            '''
            st.markdown(button_html, unsafe_allow_html=True)

    image_with_position('./static/images/announcement1.jpg', top=60, left=-120, width=350, height=204)
    image_with_position('./static/images/announcement2.jpg', top=10, left=-120, width=350, height=204)
    image_with_position('./static/images/announcement3.jpg', top=-415, left=270, width=597, height=418)

    st.write("<div style='height: 100px;'></div>", unsafe_allow_html=True)

    calendar_options = {
        "editable": "true",
        "selectable": "true",
        "headerToolbar": {
            "left": "today prev,next",
            "center": "title",
            "right": "resourceTimelineDay,resourceTimelineWeek,resourceTimelineMonth",
        },
        "slotMinTime": "08:00:00",
        "slotMaxTime": "19:00:00",
        "initialView": "resourceTimelineDay",
        "resourceGroupField": "building",
        "resources": [
            {"id": "a", "building": "EUROPE", "title": "Timisoara"},
            {"id": "b", "building": "EUROPE", "title": "Iasi"},
            {"id": "c", "building": "EUROPE", "title": "Kosice"},
            {"id": "d", "building": "EUROPE", "title": "Riga"},
            {"id": "e", "building": "EUROPE", "title": "Prague"},
            {"id": "f", "building": "ASIA", "title": "Bangalore"},
            {"id": "g", "building": "ASIA", "title": "Mumbai"},
            {"id": "h", "building": "ASIA", "title": "Pune"},
            {"id": "i", "building": "ASIA", "title": "Hyderabad"},
            {"id": "j", "building": "NORTH AMERICA", "title": "Pittsburg"},
        ],
    }
    calendar_events = [
        {
            "title": "HRBot Presentation",
            "start": "2024-06-14T14:30:00",
            "end": "2024-06-14T16:30:00",
            "resourceId": "a",
        },
        {
            "title": "HRBot Presentation",
            "start": "2024-06-14T14:30:00",
            "end": "2024-06-14T16:30:00",
            "resourceId": "b",
        },
        {
            "title": "Townhall Meeting",
            "start": "2024-06-18T10:00:00",
            "end": "2024-06-18T12:30:00",
            "resourceId": "f",
        }
    ]
    custom_css = """
        .fc-event-past {
            background-color: #0631B6;
            opacity: 0.8;
        }
        .fc-event-time {
            background-color: #0631B6;
            font-style: roboto;
        }
        .fc-event-title {
            font-weight: 700;
        }
        .fc-toolbar-title {
            font-size: 2rem;
        }
        .fc-button {
            background-color: #ED82E6;
            color: white;
        }
        .fc-button:hover {
            background-color: #ffffff 
            opacity: 0.5;
        }
    """

    calendar1 = calendar(events=calendar_events, options=calendar_options, custom_css=custom_css)
    st.write(calendar1)

    st.write("<div style='height: 50px;'></div>", unsafe_allow_html=True)

    image_with_position('./static/images/tools.jpg', top=50, left=35, width=356, height=300)
    image_with_position('./static/images/employee.jpg', top=-300, left=435, width=230, height=300)
    image_with_position('./static/images/team.jpg', top=50, left=-120, width=944, height=200)
    image_with_position('./static/images/events.jpg', top=50, left=-120, width=467, height=289)
    image_with_position('./static/images/documents1.jpg', top=-288, left=360, width=467, height=289, hyperlink="https://acrobat.adobe.com/id/urn:aaid:sc:EU:664c7751-244d-41dd-8d81-ded435838907")

if __name__ == '__main__':
    show_dashboard()
