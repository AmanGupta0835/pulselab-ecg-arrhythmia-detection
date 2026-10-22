import streamlit as st
import numpy as np, pandas as pd, matplotlib.pyplot as plt, tempfile, os
from scipy.signal import butter, filtfilt, find_peaks

st.set_page_config(page_title="PulseLab | ECG", page_icon="〰", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp{background:#f4f8fb;color:#102a43}.block-container{max-width:1250px;padding-top:1.5rem}
.hero{background:linear-gradient(120deg,#102a43,#17616b);padding:30px 34px;border-radius:20px;color:white;margin-bottom:22px}
.hero small{color:#7ee7c4;letter-spacing:2px;font-weight:700}.hero h1{font-family:'Space Grotesk';color:white;font-size:34px;margin:8px 0}.hero p{color:#d5e7f0;margin:0}
h2,h3{font-family:'Space Grotesk';color:#102a43}
.metric{background:white;border:1px solid #e0e8ef;border-radius:15px;padding:17px 20px}
.metric label{font-size:11px;color:#64748b;letter-spacing:1px;font-weight:700}.metric strong{display:block;font-size:26px;color:#102a43;margin-top:5px}
.note{background:#e8f7f0;border-left:4px solid #15966b;padding:13px;border-radius:8px;color:#14533d}
.warn{background:#fff5df;border-left:4px solid #e3a329;padding:13px;border-radius:8px;color:#744b00}
</style>
<div class="hero"><small>IPCV • DEEP LEARNING PROTOTYPE</small><h1>ECG Arrhythmia Detection</h1><p>Preprocess signals, inspect heartbeat patterns and run your trained 1D CNN.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("PulseLab")
    st.caption("ECG analysis workspace")
    fs=st.number_input("Sampling frequency (Hz)",100,2000,360,10)
    model_file=st.file_uploader("Trained model (.keras)",type=["keras"])
    st.markdown("---")
    st.markdown("**Model classes**")
    st.write("N · Normal / related")
    st.write("S · Supraventricular")
    st.write("V · Ventricular")
    st.write("F · Fusion")
    st.write("Q · Paced / other")

a,b=st.columns([1.1,.9],gap="large")
with a:
    st.subheader("01 / Upload ECG signal")
    st.caption("Upload a CSV with a numeric ECG signal column.")
    csv=st.file_uploader("ECG signal (.csv)",type=["csv"])
    signal=None
    if csv:
        try:
            df=pd.read_csv(csv)
            cols=list(df.select_dtypes(include=[np.number]).columns)
            if cols:
                col=st.selectbox("Signal column",cols)
                signal=df[col].dropna().to_numpy(float)
            else: st.error("No numeric columns found in this CSV.")
        except Exception as e: st.error(f"Could not read CSV: {e}")
    else:
        st.markdown('<div class="note"><b>Quick start</b><br>Upload a signal CSV and your trained <code>.keras</code> model to enable classification.</div>',unsafe_allow_html=True)
with b:
    st.subheader("02 / Model status")
    if model_file:
        st.markdown('<div class="note"><b>Model file selected</b><br>Ready to load for inference.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn"><b>No model loaded</b><br>Upload <code>ecg_arrhythmia_1d_cnn.keras</code> saved from your notebook.</div>',unsafe_allow_html=True)
    st.write("")
    st.markdown("**Pipeline**")
    st.code("ECG CSV → Filter → Segment → 1D CNN → Prediction",language="text")

if signal is not None and len(signal)>30:
    nyq=fs/2
    bb,aa=butter(3,[0.5/nyq,min(40,nyq*.9)/nyq],btype="band")
    clean=filtfilt(bb,aa,signal) if len(signal)>30 else signal
    st.subheader("03 / Signal preview")
    fig,ax=plt.subplots(figsize=(12,3.2))
    t=np.arange(len(signal))/fs
    ax.plot(t,signal,color="#dc5967",alpha=.6,lw=.8,label="Raw")
    ax.plot(t,clean,color="#15966b",lw=1,label="Filtered")
    ax.set_xlabel("Time (seconds)");ax.set_ylabel("Amplitude");ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.patch.set_alpha(0);st.pyplot(fig,use_container_width=True);plt.close(fig)
    c1,c2,c3=st.columns(3)
    for container,label,value in [(c1,"SAMPLES",f"{len(signal):,}"),(c2,"DURATION",f"{len(signal)/fs:.1f}s"),(c3,"FILTER","0.5–40 Hz")]:
        with container: st.markdown(f'<div class="metric"><label>{label}</label><strong>{value}</strong></div>',unsafe_allow_html=True)
    st.subheader("04 / Beat segmentation & prediction")
    st.markdown('<div class="warn">Automatic peak detection is approximate. Use the notebook’s held-out test-set metrics for your reported F1-score. This is an educational demo, not a medical tool.</div>',unsafe_allow_html=True)
    if st.button("Analyze signal",type="primary",use_container_width=True):
        peaks,_=find_peaks(clean,distance=int(.28*fs),prominence=max(np.std(clean)*.45,1e-6))
        before,after=int(.20*fs),int(.40*fs)
        segments=[];times=[]
        for p in peaks:
            if p-before>=0 and p+after<=len(clean):
                seg=clean[p-before:p+after].astype(np.float32);sd=seg.std()
                if sd>1e-6 and len(seg)==216:
                    segments.append((seg-seg.mean())/sd);times.append(p/fs)
        st.write(f"Candidate heartbeat segments: **{len(segments)}**")
        if not segments: st.warning("No suitable segments found. Try a longer signal or verify the sampling frequency.")
        elif not model_file: st.warning("Upload the trained .keras model to generate predictions.")
        else:
            try:
                import tensorflow as tf
                with tempfile.NamedTemporaryFile(suffix=".keras",delete=False) as f:
                    f.write(model_file.getbuffer());model_path=f.name
                model=tf.keras.models.load_model(model_path)
                pred=model.predict(np.asarray(segments)[...,None],verbose=0)
                labels=["N","S","V","F","Q"];ids=np.argmax(pred,axis=1)
                out=pd.DataFrame({"Beat":np.arange(1,len(ids)+1),"Time (s)":np.round(times,3),"Predicted class":[labels[i] for i in ids],"Confidence (%)":np.round(np.max(pred,axis=1)*100,1)})
                st.dataframe(out,use_container_width=True,hide_index=True)
                counts=out["Predicted class"].value_counts().reindex(labels,fill_value=0)
                fig2,ax2=plt.subplots(figsize=(8,3));ax2.bar(counts.index,counts.values,color="#15966b")
                ax2.set_title("Predicted heartbeat classes");ax2.set_ylabel("Count");ax2.grid(axis="y",alpha=.2)
                st.pyplot(fig2,use_container_width=True);plt.close(fig2)
                st.download_button("Download predictions CSV",out.to_csv(index=False),"ecg_predictions.csv","text/csv")
            except Exception as e: st.error(f"Prediction failed: {e}")
else:
    st.subheader("03 / Signal preview")
    st.markdown('<div class="metric"><label>READY FOR YOUR DATA</label><strong>Upload an ECG CSV</strong><span> The waveform and analysis panels will appear here.</span></div>',unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center><small>PulseLab · ECG Arrhythmia Detection · IPCV student prototype<br>For educational demonstration only — not for diagnosis.</small></center>",unsafe_allow_html=True)
