import streamlit as st
import plotly.graph_objects as go
import json
import plotly.express as px

def load_metrics():
    with open("/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_WINE/metrics.json", "r") as f:
        return json.load(f)

def load_report():
    with open("/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_WINE/classification_report.txt", "r") as f:
        return f.read()

def model_page():
    st.title("Modelo")
    st.write("Resultados y métricas del modelo:")

    # Load metrics
    metrics = load_metrics()

    # Mostrar ROC Curve para cada clase
    st.subheader("ROC Curve por Clase")
    fig_roc = go.Figure()
    for i in metrics['roc_auc']:
        if i == "micro":
            continue
        fig_roc.add_trace(go.Scatter(x=metrics['fpr'][i], y=metrics['tpr'][i],
                                     mode='lines', name=f'Clase {i} (AUC = {metrics["roc_auc"][i]:.2f})'))
    fig_roc.add_trace(go.Scatter(x=metrics['fpr']['micro'], y=metrics['tpr']['micro'],
                                 mode='lines', name=f'Micro-averaged (AUC = {metrics["roc_auc"]["micro"]:.2f})'))
    fig_roc.update_layout(title='Receiver Operating Characteristic por Clase',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate')
    st.plotly_chart(fig_roc)

    # Mostrar Precision-Recall Curve
    st.subheader("Precision-Recall Curve por Clase")
    fig_pr = go.Figure()
    for i in metrics['precision']:
        if i == "micro":
            continue
        fig_pr.add_trace(go.Scatter(x=metrics['recall'][i], y=metrics['precision'][i],
                                    mode='lines', name=f'Clase {i}'))
    fig_pr.add_trace(go.Scatter(x=metrics['recall']['micro'], y=metrics['precision']['micro'],
                                mode='lines', name='Micro-averaged'))
    fig_pr.update_layout(title='Precision-Recall Curve por Clase',
                         xaxis_title='Recall',
                         yaxis_title='Precision')
    st.plotly_chart(fig_pr)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = metrics['confusion_matrix']
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       labels={'x': 'Predicted', 'y': 'Actual'})
    fig_cm.update_layout(title='Confusion Matrix')
    st.plotly_chart(fig_cm)

    # Classification Report
    st.subheader("Classification Report")
    st.text(load_report())

    # Download Report
    st.subheader("Descargar Informe")
    with open("/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_WINE/classification_report.txt", "r") as f:
        report = f.read()
    st.download_button(label="Descargar Informe", data=report, file_name="classification_report.txt")
