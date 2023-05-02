# coding=utf-8
import argparse
import utils
import commons
import torch
import gradio as gr
import webbrowser
from models import SynthesizerTrn
from text import text_to_sequence
from torch import no_grad, LongTensor
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

def get_text(text, hps):
    text_norm= text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(net_g_ms):
    def tts_fn(text, noise_scale, noise_scale_w, length_scale, speaker_id):
        text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
        stn_tst= get_text(text, hps_ms)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                   length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
        return "Success", (22050, audio)
    return tts_fn


download_audio_js = """
() =>{{
    let root = document.querySelector("body > gradio-app");
    if (root.shadowRoot != null)
        root = root.shadowRoot;
    let audio = root.querySelector("#tts-audio").querySelector("audio");
    let text = root.querySelector("#input-text").querySelector("textarea");
    if (audio == undefined)
        return;
    text = text.value;
    if (text == undefined)
        text = Math.floor(Math.random()*100000000);
    audio = audio.src;
    let oA = document.createElement("a");
    oA.download = text.substr(0, 20)+'.wav';
    oA.href = audio;
    document.body.appendChild(oA);
    oA.click();
    oA.remove();
}}
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--colab", action="store_true", default=False)
    parser.add_argument('-c', '--config', type=str, default="configs/config.json", help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,  help='Model path')
    args = parser.parse_args()
    device = torch.device(args.device)
    hps_ms = utils.get_hparams_from_file(args.config)
    models = []
    net_g_ms = SynthesizerTrn(
        len(hps_ms.symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model)
    utils.load_checkpoint(args.model, net_g_ms, None)
    _ = net_g_ms.eval().to(device)
    models.append((net_g_ms, create_tts_fn(net_g_ms,)))
    with gr.Blocks() as app:
        with gr.Tabs():
            for (net_g_ms, tts_fn) in models:
                with gr.TabItem(args.model):
                    with gr.Row():
                        with gr.Column():
                            input_text = gr.Textbox(label="Text",
                                                    lines=5, value="今日はいい天気ですね。",
                                                    elem_id=f"input-text")
                            btn = gr.Button(value="Generate", variant="primary")
                            sid = gr.Number(label="speaker_id", value=10)
                            with gr.Row():
                                ns = gr.Slider(label="noise_scale", minimum=0.1, maximum=1.0, step=0.1, value=0.6,
                                               interactive=True)
                                nsw = gr.Slider(label="noise_scale_w", minimum=0.1, maximum=1.0, step=0.1, value=0.668,
                                                interactive=True)
                                ls = gr.Slider(label="length_scale", minimum=0.1, maximum=2.0, step=0.1, value=1.0)
                        with gr.Column():
                            o1 = gr.Textbox(label="Output Message")
                            o2 = gr.Audio(label="Output Audio", elem_id=f"tts-audio")
                            download = gr.Button("Download Audio")
                        btn.click(tts_fn, inputs=[input_text, ns, nsw, ls, sid], outputs=[o1, o2], api_name=f"tts")
                        download.click(None, [], [], _js=download_audio_js)
    if args.colab:
        webbrowser.open("http://127.0.0.1:7860")
    app.queue(concurrency_count=1, api_open=args.api).launch(share=args.share)
