import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from prophet.make_holidays import make_holidays_df
from datetime import datetime
from scipy import stats  

st.set_page_config(layout="wide", initial_sidebar_state="expanded")


def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_color()


app_modes = [
    "Data Simulation",
    "Exploratory Data Analysis (EDA)",
    "Hotspot Analysis",
    "Foot Traffic Forecasting",
    "Business Insights Report"
]


if 'app_mode_index' not in st.session_state:
    st.session_state['app_mode_index'] = 0


def next_app_mode():
    if st.session_state['app_mode_index'] < len(app_modes) - 1:
        st.session_state['app_mode_index'] += 1

def prev_app_mode():
    if st.session_state['app_mode_index'] > 0:
        st.session_state['app_mode_index'] -= 1


def navigation_buttons():
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.button('Back', on_click=prev_app_mode, disabled=(st.session_state['app_mode_index'] == 0))
    with col3:
        st.button('Next', on_click=next_app_mode, disabled=(st.session_state['app_mode_index'] == len(app_modes) - 1))


def data_simulation():
    st.header("1. Upload Your Own Data or Simulate Data")


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader("Upload a CSV file with 'date' and 'foot_traffic' columns", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, parse_dates=['date'])
            st.success("Data loaded successfully!")
            st.session_state['df'] = df

    with col2:
        st.subheader("Simulate Data")
        if st.button("Simulate Data"):
          
            np.random.seed(42)
            dates = pd.date_range(start='2023-01-01', periods=180, freq='D')
            foot_traffic = (
                200
                + 50 * np.sin(2 * np.pi * dates.dayofyear / 365)
                + 30 * np.sin(2 * np.pi * dates.dayofweek / 7)
                + np.random.normal(0, 20, len(dates))
            )
            foot_traffic = np.maximum(foot_traffic, 0)
            df = pd.DataFrame({
                'date': dates,
                'foot_traffic': foot_traffic.astype(int)
            })
            st.success("Simulated data created!")
            st.session_state['df'] = df

    if 'df' in st.session_state:
        st.subheader("Sample of the Data")
        st.write(st.session_state['df'].head())

    navigation_buttons()


def exploratory_data_analysis():
    if 'df' not in st.session_state:
        st.warning("Please upload or simulate data first in the 'Data Simulation' section.")
    else:
        df = st.session_state['df']

       
        col1, col2 = st.columns(2)

        with col1:
            
            st.subheader("Foot Traffic Over Time")
            fig = px.line(df, x='date', y='foot_traffic', title='Daily Foot Traffic',
                          labels={'foot_traffic': 'Foot Traffic', 'date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)

           
            st.subheader("Summary Statistics")
            summary_stats = df['foot_traffic'].describe()
            st.table(summary_stats)

        with col2:
           
            st.subheader("Distribution of Foot Traffic")

            
            fig2 = go.Figure()

          
            fig2.add_trace(go.Histogram(
                x=df['foot_traffic'],
                nbinsx=30,  
                histnorm='probability density',
                marker_color='rgba(0, 128, 128, 0.6)',  
                name='Histogram'
            ))

           
            x_values = np.linspace(df['foot_traffic'].min(), df['foot_traffic'].max(), 1000)
            kde = stats.gaussian_kde(df['foot_traffic'])
            fig2.add_trace(go.Scatter(
                x=x_values,
                y=kde(x_values),
                mode='lines',
                name='Density Curve',
                line=dict(color='darkblue', width=2)
            ))

         
            fig2.update_layout(
                title='Distribution of Foot Traffic',
                xaxis_title='Foot Traffic',
                yaxis_title='Probability Density',
                bargap=0.05,
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.01,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=40, r=40, t=60, b=40)
            )

           
            st.plotly_chart(fig2, use_container_width=True)

        
        st.subheader("Average Foot Traffic by Day of Week")
        df['day_of_week'] = df['date'].dt.day_name()
        avg_traffic_weekday = df.groupby('day_of_week')['foot_traffic'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig3 = px.bar(x=avg_traffic_weekday.index, y=avg_traffic_weekday.values,
                      title='Average Foot Traffic by Day of Week',
                      labels={'x': 'Day of Week', 'y': 'Average Foot Traffic'})
        st.plotly_chart(fig3, use_container_width=True)

    navigation_buttons()


def hotspot_analysis():
    st.markdown("""
    # Hotspot Analysis

    This section performs clustering on spatial customer data to identify hotspots within the retail space. The store layout, including walls, is displayed for context.
    """)

    st.header("Upload Customer Position Data")

    st.markdown("""
    Please upload a CSV file containing customer positions with **'x'** and **'y'** columns.
    The **'x'** and **'y'** values should represent the coordinates of customers within the store layout.

    **Or**, enter the number of customers to simulate and click the button below to generate customer position data.
    """)

    positions = None

    
    uploaded_file = st.file_uploader("Upload a CSV file with 'x' and 'y' columns", type=["csv"])
    if uploaded_file is not None:
        try:
            positions = pd.read_csv(uploaded_file)
            if {'x', 'y'}.issubset(positions.columns):
                st.success("Customer positions data loaded successfully!")
                st.write(positions.head())
                st.session_state['positions'] = positions
            else:
                st.error("The uploaded CSV file must contain 'x' and 'y' columns.")
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

    
    st.subheader("Or Simulate Customer Positions")
    num_customers = st.number_input(
        "Enter the number of customers to simulate",
        min_value=1,
        value=100,
        step=1,
        key='num_customers'
    )
    if st.button("Simulate Data"):
        positions = pd.DataFrame({
            'x': np.random.uniform(0, 100, int(num_customers)),
            'y': np.random.uniform(0, 100, int(num_customers))
        })
        st.success("Simulated customer positions created!")
        st.write(positions.head())
        st.session_state['positions'] = positions

    if 'positions' in st.session_state:
        positions = st.session_state['positions']

        st.header("Define Store Layout (Walls)")
        st.markdown("""
        You can either use a predefined store layout or upload a file containing the store's wall coordinates.
        """)

     
        layout_option = st.selectbox("Choose store layout option", ["Default Layout", "Upload Layout File"])

        if layout_option == "Default Layout":
            
            walls = [
                {'x0': 0, 'y0': 0, 'x1': 100, 'y1': 0},     
                {'x0': 100, 'y0': 0, 'x1': 100, 'y1': 100}, 
                {'x0': 100, 'y0': 100, 'x1': 0, 'y1': 100},  
                {'x0': 0, 'y0': 100, 'x1': 0, 'y1': 0},   
          
                {'x0': 50, 'y0': 0, 'x1': 50, 'y1': 50},
                {'x0': 50, 'y0': 50, 'x1': 100, 'y1': 50}
            ]
            st.success("Using default store layout.")
        else:
            
            st.subheader("Upload Store Layout File")
            st.markdown("""
            Please upload a CSV file containing wall coordinates with columns: **'x0'**, **'y0'**, **'x1'**, **'y1'**.
            Each row represents a wall segment from point (**x0**, **y0**) to point (**x1**, **y1**).
            """)
            layout_file = st.file_uploader("Upload a CSV file with wall coordinates", type=["csv"], key='layout_file')
            if layout_file is not None:
                try:
                    walls_df = pd.read_csv(layout_file)
                    if {'x0', 'y0', 'x1', 'y1'}.issubset(walls_df.columns):
                        walls = walls_df.to_dict('records')
                        st.success("Store layout loaded successfully!")
                        st.write(walls_df.head())
                    else:
                        st.error("The uploaded CSV file must contain 'x0', 'y0', 'x1', 'y1' columns.")
                        walls = []
                except Exception as e:
                    st.error(f"An error occurred while reading the file: {e}")
                    walls = []
            else:
                st.warning("Please upload a CSV file containing wall coordinates.")
                walls = []

     
        st.subheader("Customer Positions with Store Layout")
        fig = go.Figure()

     
        fig.add_trace(go.Scatter(
            x=positions['x'], y=positions['y'],
            mode='markers',
            name='Customers',
            marker=dict(color='blue', size=5, opacity=0.5)
        ))

      
        for wall in walls:
            fig.add_shape(
                type="line",
                x0=wall['x0'], y0=wall['y0'],
                x1=wall['x1'], y1=wall['y1'],
                line=dict(color="black", width=3),
            )

        fig.update_layout(
            title='Customer Positions with Store Layout',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            showlegend=False,
            width=800, height=600
        )
        st.plotly_chart(fig)

        st.subheader("Clustering and Hotspot Identification")

  
        optimal_k = st.number_input(
            "Enter the number of clusters (k) for KMeans clustering",
            min_value=1,
            value=4,
            step=1,
            key='optimal_k'
        )
        kmeans = KMeans(n_clusters=int(optimal_k), random_state=42)
        positions['cluster'] = kmeans.fit_predict(positions[['x', 'y']])

      
        fig_clustered = px.scatter(positions, x='x', y='y', color=positions['cluster'].astype(str),
                                   title='Customer Clusters',
                                   labels={'x': 'X Position', 'y': 'Y Position', 'color': 'Cluster'})
      
        for wall in walls:
            fig_clustered.add_shape(
                type="line",
                x0=wall['x0'], y0=wall['y0'],
                x1=wall['x1'], y1=wall['y1'],
                line=dict(color="black", width=3),
            )
        fig_clustered.update_layout(width=800, height=600)
        st.plotly_chart(fig_clustered)

       
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Clustered Customer Positions")
            st.plotly_chart(fig_clustered, use_container_width=True)

        with col2:
            st.subheader("Cluster Analysis")
            
            cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['x', 'y'])
            cluster_counts = positions['cluster'].value_counts().sort_index()
            cluster_summary = pd.DataFrame({
                'Cluster': cluster_counts.index,
                'Number of Customers': cluster_counts.values,
                'Center X': cluster_centers['x'],
                'Center Y': cluster_centers['y']
            })
            st.table(cluster_summary)

            st.markdown("""
            **Insights:**

            - The clusters represent areas in the store where customers tend to gather.
            - Cluster centers indicate the hotspots within the store layout.
            - Higher customer counts in a cluster suggest more popular areas.
            """)

        
        st.session_state['positions'] = positions
        st.session_state['cluster_summary'] = cluster_summary
        st.session_state['fig_clustered'] = fig_clustered

    else:
        st.info("Either upload customer data or simulate data to proceed.")

    navigation_buttons()


def foot_traffic_forecasting():
    if 'df' not in st.session_state:
        st.warning("Please upload or simulate data first in the 'Data Simulation' section.")
    else:
        df = st.session_state['df']

        
        st.subheader("Forecast Parameters")
        periods_input = st.number_input('Number of days to forecast into the future:', min_value=1, max_value=365, value=30)

       
        st.subheader("Include Holidays in the Model")
        holiday_option = st.radio(
            "Do you want to include holidays in the model?",
            ('No', 'Use Predefined Holidays', 'Input Custom Holidays', 'Upload Holiday File')
        )

        if holiday_option == 'Use Predefined Holidays':
           
            years = [df['date'].dt.year.min(), df['date'].dt.year.max() + 1]
            holidays = make_holidays_df(year_list=list(range(years[0], years[1] + 1)), country='US')
            st.success("Predefined holidays included in the model.")
        elif holiday_option == 'Input Custom Holidays':
            st.markdown("### Input Custom Holiday Dates and Names")
            num_holidays = st.number_input('How many holidays do you want to input?', min_value=1, max_value=20, value=5)
            holiday_dates = []
            for i in range(int(num_holidays)):
                cols = st.columns(2)
                with cols[0]:
                    date = st.date_input(f'Holiday Date {i+1}', value=None, key=f'holiday_date_{i}')
                with cols[1]:
                    name = st.text_input(f'Holiday Name {i+1}', value=f'Holiday_{i+1}', key=f'holiday_name_{i}')
                if date:
                    holiday_dates.append({'ds': pd.to_datetime(date), 'holiday': name})
            if holiday_dates:
                holidays = pd.DataFrame(holiday_dates)
                st.success(f"{len(holiday_dates)} custom holidays added.")
                st.write(holidays)
            else:
                st.warning("No holidays have been added.")
                holidays = None
        elif holiday_option == 'Upload Holiday File':
            st.markdown("### Upload a CSV File with Holiday Dates")
            uploaded_holiday_file = st.file_uploader("Upload a CSV file with 'ds' and 'holiday' columns", type=["csv"])
            if uploaded_holiday_file is not None:
                holidays = pd.read_csv(uploaded_holiday_file, parse_dates=['ds'])
                if 'holiday' not in holidays.columns:
                    holidays['holiday'] = 'custom_holiday'
                st.success("Holidays from the uploaded file included.")
                st.write(holidays)
            else:
                st.warning("Please upload a CSV file with holiday dates.")
                holidays = None
        else:
            holidays = None

        
        st.subheader("Model Training and Evaluation")
        test_size = st.slider('Number of days in test set:', min_value=7, max_value=90, value=30, step=7)
        df_train = df.iloc[:-test_size]
        df_test = df.iloc[-test_size:]

 
        df_train_prophet = df_train.rename(columns={'date': 'ds', 'foot_traffic': 'y'})
        df_test_prophet = df_test.rename(columns={'date': 'ds', 'foot_traffic': 'y'})

       
        with st.spinner('Training the forecasting model...'):
            model = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(df_train_prophet)

     
        future = model.make_future_dataframe(periods=test_size + periods_input)
        forecast = model.predict(future)

      
        st.subheader("Model Performance on Test Data")
        forecast_test = forecast[forecast['ds'].isin(df_test_prophet['ds'])]
        forecast_test = forecast_test.merge(df_test_prophet[['ds', 'y']], on='ds')
        forecast_test['error'] = forecast_test['y'] - forecast_test['yhat']
        forecast_test['abs_error'] = forecast_test['error'].abs()
        mape = np.mean(forecast_test['abs_error']/forecast_test['y'])*100
        rmse = np.sqrt(np.mean(forecast_test['error']**2))

       
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h3 style='text-align: center;'>Performance Metrics</h3>", unsafe_allow_html=True)
            st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

           
            st.subheader("Actual vs. Predicted Foot Traffic")
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(
                x=forecast_test['ds'],
                y=forecast_test['y'],
                mode='lines+markers',
                name='Actual'
            ))
            fig_compare.add_trace(go.Scatter(
                x=forecast_test['ds'],
                y=forecast_test['yhat'],
                mode='lines+markers',
                name='Predicted'
            ))
            fig_compare.update_layout(
                title='Actual vs. Predicted Foot Traffic',
                xaxis_title='Date',
                yaxis_title='Foot Traffic'
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        with col2:
           
            st.subheader('Forecast Components')
            components_fig = model.plot_components(forecast)
            st.pyplot(components_fig)
          
            buf = BytesIO()
            components_fig.savefig(buf, format='png')
            st.session_state['forecast_components_image'] = buf

      
        st.subheader("Future Forecast")
        forecast_future = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-periods_input:].copy()
        forecast_future = forecast_future.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Foot Traffic',
                                                          'yhat_lower': 'Lower Confidence Interval',
                                                          'yhat_upper': 'Upper Confidence Interval'})
        st.write(forecast_future)

       
        fig_future = go.Figure()

        
        fig_future.add_trace(go.Scatter(
            x=forecast_future['Date'],
            y=forecast_future['Forecasted Foot Traffic'],
            mode='lines',
            name='Forecasted Foot Traffic',
            line=dict(color='blue')
        ))

        
        fig_future.add_trace(go.Scatter(
            x=forecast_future['Date'].tolist() + forecast_future['Date'][::-1].tolist(),
            y=forecast_future['Upper Confidence Interval'].tolist() + forecast_future['Lower Confidence Interval'][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(173,216,230,0.2)', 
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))

        fig_future.update_layout(
            title='Future Foot Traffic Forecast',
            xaxis_title='Date',
            yaxis_title='Foot Traffic',
            template='plotly_white'
        )

        st.plotly_chart(fig_future, use_container_width=True)

        st.markdown("""
        **Insights:**

        - The model's performance on the test data indicates its ability to generalize to unseen data.
        - Including holidays may improve the model's accuracy if holidays significantly impact foot traffic.
        - The components plot reveals patterns in the data, such as trends, weekly seasonality, and the effect of holidays.
        """)

       
        st.session_state['forecast'] = forecast
        st.session_state['forecast_future'] = forecast_future
        st.session_state['fig_future'] = fig_future
        st.session_state['periods_input'] = periods_input

    navigation_buttons()


def business_insights_report():
    if 'df' not in st.session_state:
        st.warning("Please complete the previous sections to generate the report.")
    else:
        df = st.session_state['df']
        df['date'] = pd.to_datetime(df['date'])

        
        def generate_report_html():
            import base64

            report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

          
            html_parts = []
            html_parts.append(f"""
            <html>
            <head>
                <title>Business Insights Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2, h3 {{ color: #2E4053; }}
                    p {{ font-size: 14px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    ul {{ list-style-type: disc; margin-left: 20px; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
            <h1>Business Insights Report</h1>
            <p><strong>Generated on:</strong> {report_date}</p>
            """)

          
            html_parts.append("<h2>Foot Traffic Forecast</h2>")

            if 'forecast_future' in st.session_state and 'fig_future' in st.session_state:
                forecast_future = st.session_state['forecast_future']
                fig_future = st.session_state['fig_future']
                periods_input = st.session_state.get('periods_input', 30)

              
                fig_future_html = fig_future.to_html(include_plotlyjs='cdn', full_html=False)
                html_parts.append("<h3>Forecasted Foot Traffic</h3>")
                html_parts.append(fig_future_html)

              
                forecast_future_html = forecast_future.to_html(index=False)
                html_parts.append("<h3>Forecast Data</h3>")
                html_parts.append(forecast_future_html)

             
                if 'forecast_components_image' in st.session_state:
                    buf = st.session_state['forecast_components_image']
                    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                    html_parts.append("<h3>Forecast Components</h3>")
                    html_parts.append(f'<img src="data:image/png;base64,{base64_img}" alt="Forecast Components">')
                else:
                    html_parts.append("<p><em>Forecast components not available. Please complete the Foot Traffic Forecasting section.</em></p>")
            else:
                html_parts.append("<p><em>Forecast data not available. Please complete the Foot Traffic Forecasting section.</em></p>")

            
            html_parts.append("""
            </body>
            </html>
            """)

           
            html_report = ''.join(html_parts)

            return html_report

        
        html_report = generate_report_html()

        
        st.components.v1.html(html_report, height=800, scrolling=True)

        
        st.subheader("Download Report")
        st.write("Click the button below to download the report as an HTML file.")
        st.download_button(
            label="Download Report as HTML",
            data=html_report,
            file_name='business_insights_report.html',
            mime='text/html'
        )

    navigation_buttons()


def main():
   
    app_mode = app_modes[st.session_state['app_mode_index']]

    
    if app_mode == "Data Simulation":
        data_simulation()
    elif app_mode == "Exploratory Data Analysis (EDA)":
        exploratory_data_analysis()
    elif app_mode == "Hotspot Analysis":
        hotspot_analysis()
    elif app_mode == "Foot Traffic Forecasting":
        foot_traffic_forecasting()
    elif app_mode == "Business Insights Report":
        business_insights_report()

if __name__ == "__main__":
    main()
