import streamlit as st
import base64

def show_profiles():
    def set_page_bg_color(color):
        page_bg_style = f'''
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        '''
        st.markdown(page_bg_style, unsafe_allow_html=True)

    set_page_bg_color("#E9E9E9")

    @st.cache_data
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    
    def image_with_position_border(png_file, top, left, width, height, style, radius, color, b, hyperlink=None):
        bin_str = get_base64_of_bin_file(png_file)
        image_html = f'''
        <style>
        .bordered-image {{
            border-style: {style};
            border-radius: {radius}px;
            border-color:  {color};
            border-width: {b}px;
            width: {width}px;
            height: {height}px;
            margin-top: {top}px;
            margin-left: {left}px;
        }}
        </style>
        <div>
            <img src="data:image/png;base64,{bin_str}" class="bordered-image">
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

    left_div_style = '''
    <style>
    .left-div {
        height: 100%;
        width: 100px;
        background-color: #ED82E6;
    }
    </style>
    '''

    st.markdown(left_div_style, unsafe_allow_html=True)

    st.markdown("<div style='margin-left: -300px; font-size: 2em; color: black; font-weight: 600'>HRBot Settings</div>", unsafe_allow_html=True)
    st.write("<div style='height: 75px;'></div>", unsafe_allow_html=True)
    st.write("<div style='margin-top: -30px; margin-left: -150px; font-size: 1.5em; color: #4F4F4F'>Profile</div>", unsafe_allow_html=True)
    st.write("<div style='margin-left: -150px; font-size: 2em; color: #4F4F4F'>HR Department</div>", unsafe_allow_html=True)

    def set_page_bg_color(color):
        page_bg_style = f'''
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        '''
        st.markdown(page_bg_style, unsafe_allow_html=True)

    set_page_bg_color("#E9E9E9")

    image_with_position('./static/images/profile.jpg',top=-110, left=-300, width=96, height=96)
    image_with_position('./static/images/profile2.jpg',top=-125 , left=-225, width=32, height=32)
    image_with_position_border('./static/images/team-members.jpg',top=70, left=-30, width=800, height=411, style="solid", radius=5, color="#000000", b=2)

if __name__ == '__main__':
    show_profiles()