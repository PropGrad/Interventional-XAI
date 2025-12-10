import gradio as gr
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch as T
from torch.utils import data
from torchvision import transforms
import diffusers
from pathlib import Path
from tqdm import tqdm
from celluloid import Camera
from propgrad.utils import subsample_plot, ImageList_DS, perform_inference
from propgrad.convmixer import ConvMixer
from propgrad.zollstock import Zollstock
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams.update(
    {
        'text.usetex': True,
        "font.family": "serif",
        "font.size": 22,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
    }
)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amssymb}\usepackage{wasysym}')
warnings.simplefilter("ignore")


def load_cm(path, verbose=False):
    load_dict = T.load(path)

    M = ConvMixer(**load_dict["paras"])
    M.load_state_dict(load_dict["state_dict"])

    M.eval()

    if verbose:
        print("ACC:", load_dict["performance"]["acc"])
        print("Loss:", load_dict["performance"]["loss"])

    return M, load_dict["performance"]["acc"]


class Interventional_Explainer:
    def __init__(self, ):
        # we overimplement some of our utilities 
        # so we don't have to load the model multiple times
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
            'timbrooks/instruct-pix2pix', torch_dtype=T.float16, safety_checker=None).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.interventional_data = []

        self.test_transform = transforms.Compose([
            # Resize the images to IMAGE_SIZE xIMAGE_SIZE
            transforms.Resize(size=(128, 128)),
            # Turn the image into a torch.Tensor
            # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
            transforms.ToTensor(),
            # normalize to [-1,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # loading models for later explanation
        print("Unbiased:")
        self.M_unb, _ = load_cm("checkpoints/cats-vs-dogs/convmixer_unbiased.pth", verbose=True)
        print("\nCats have dark fur bias:")
        self.M_dcats, _ = load_cm("checkpoints/cats-vs-dogs/convmixer_dark-cats-bias.pth", verbose=True)
        print("\nDogs have dark fur bias:")
        self.M_ddogs, _ = load_cm("checkpoints/cats-vs-dogs/convmixer_dark-dogs-bias.pth", verbose=True)


        self.Z = Zollstock()



    def explain(self, input_image):
        if len(self.interventional_data) == 0:
            gr.Warning("No interventional data found! Please generate interventions first.")
            return None
        
        if isinstance(input_image, str) or isinstance(input_image, Path):
            input_image = Image.open(input_image).convert('RGB')
        elif isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image.astype('uint8')).convert('RGB')
        
        D = ImageList_DS([input_image] + self.interventional_data, self.test_transform)
        loader = data.DataLoader(D, batch_size=10, shuffle=False)
        
        # predicting the models
        logits_unb = perform_inference(self.M_unb, loader)
        logits_dc = perform_inference(self.M_dcats, loader)
        logits_dd = perform_inference(self.M_ddogs, loader)

        propgrad_unb, pval_unb, null_dist_unb = self.Z.shuffle_test(logits_unb[:,0], plot_null_dist=True)
        null_dist_unb.savefig("tmp/null_unb.png", transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(null_dist_unb)
        
        propgrad_dc, pval_dc, null_dist_dc = self.Z.shuffle_test(logits_dc[:,0], plot_null_dist=True)
        null_dist_dc.savefig("tmp/null_dcats.png", transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(null_dist_dc)

        propgrad_dd, pval_dd, null_dist_dd = self.Z.shuffle_test(logits_dd[:,0], plot_null_dist=True)
        null_dist_dd.savefig("tmp/null_ddogs.png", transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(null_dist_dd)

        print(f"Unbiased PropGrad: {propgrad_unb:.4f}, p-value: {pval_unb:.4f}")
        print(f"Dark Cats Bias PropGrad: {propgrad_dc:.4f}, p-value: {pval_dc:.4f}")
        print(f"Dark Dogs Bias PropGrad: {propgrad_dd:.4f}, p-value: {pval_dd:.4f}")
        
        fig = plt.figure(figsize=(14,4.5))
        ax = fig.add_subplot(111)

        # order should be cats, dogs
        me = max(1, (len(self.interventional_data)+1) // 7)
        lw = 3
        ms = 12

        c_unb = "#f72585" 
        c_dcats = "#3a0ca3" 
        c_ddogs = "#4895ef" 

        num = len(self.interventional_data) +1
        vals = np.arange(num)

        ax.plot(vals, logits_unb[:,0], label="Unbiased\n" + r"$\mathbb{E}[|\nabla_\mathsf{X}\mathbb{F}|] = " + f"{propgrad_unb:.3f}$\n" + f"$p$-val = {pval_unb:.3f}", color=c_unb, linestyle="-", linewidth=lw, )
        ax.plot(vals, logits_dc[:,0], label="Dark Cats Bias\n" + r"$\mathbb{E}[|\nabla_\mathsf{X}\mathbb{F}|] = " + f"{propgrad_dc:.3f}$\n" + f"$p$-val = {pval_dc:.3f}", color=c_dcats, linestyle=":", linewidth=lw, marker="*", markevery=me, markersize=ms)
        ax.plot(vals, logits_dd[:,0], label="Dark Dogs Bias\n" + r"$\mathbb{E}[|\nabla_\mathsf{X}\mathbb{F}|] = " + f"{propgrad_dd:.3f}$\n" + f"$p$-val = {pval_dd:.3f}", color=c_ddogs, linestyle=":", linewidth=lw, marker="v", markevery=me, markersize=ms)
        ax.axhline(0.5, color="r", linestyle="--")

        ax.set_ylim((-0.1,1.1))

        ax.set_xticks([num*0.05,(num-1)*0.95], ["Original", "Interventional"])
        ax.set_yticks([0,1], ["Dog", "Cat"])


        ax.set_ylabel("Prediction")
        ax.set_xlabel("Data")
        ax.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), title="ConvMixer Models",
                ncol=1, fancybox=True, shadow=True) 

        fig.tight_layout(pad=0.2)
        if not Path("tmp").exists():
            Path("tmp").mkdir(parents=True, exist_ok=True)

        fig.savefig("tmp/explanation.png", transparent=False, bbox_inches='tight', pad_inches=0.2)
        plt.close('all')

        return gr.update(value="tmp/explanation.png", visible=True), gr.update(value="tmp/null_unb.png", visible=True), gr.update(value="tmp/null_dcats.png", visible=True), gr.update(value="tmp/null_ddogs.png", visible=True)
    


    def generate_interventions(self, input_image, instruction, cfg_img, max_cfg_scale, num_steps, seed):
        # First, actual generation
        cfg_txt = np.linspace(1.01, max_cfg_scale, int(num_steps), endpoint=True)
        if isinstance(input_image, str) or isinstance(input_image, Path):
            input_image = Image.open(input_image).convert('RGB')
        elif isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image.astype('uint8')).convert('RGB')
        self.interventional_data = []
        with T.inference_mode():
            for cfg_t in tqdm(cfg_txt):
                res = self.pipe(prompt=instruction, image=input_image, generator=T.Generator(device=self.device).manual_seed(seed), guidance_scale=cfg_t,
                            image_guidance_scale=cfg_img).images[0]
                self.interventional_data.append(res)

        # next some visualizations for gradio
        if not Path("tmp").exists():
            Path("tmp").mkdir(parents=True, exist_ok=True)

        fig = subsample_plot(self.interventional_data, num=min(num_steps, 7), s=2)
        fig.savefig("tmp/interventions.png", transparent=True, bbox_inches='tight', pad_inches=0)
        
        fig = plt.figure(frameon=False)
        fig.set_size_inches(4,4)
        fig.patch.set_alpha(0.)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        camera = Camera(fig)
        for img in self.interventional_data:
            ax.imshow(img, aspect='auto')
            fig.tight_layout()
            camera.snap()
        ani = camera.animate(interval=100, repeat=True)
        ani.save("tmp/interventions.mp4", writer=animation.FFMpegWriter(fps=5))

        plt.close('all')
        return "tmp/interventions.png", "tmp/interventions.mp4"
    


if __name__ == "__main__":
    explainer = Interventional_Explainer()

    demo = gr.Blocks()
    with demo:
        with gr.Row():   
            with gr.Column(scale=2, min_width=300):
                with gr.Row():
                    input_image = gr.Image("example_images/dog_001.jpg", height=300, width=400, label="Input Image", interactive=True)
                with gr.Row():
                    instruction = gr.Textbox(value="Change the fur color to black. Leave other details the same.", max_lines=3, label="Intervention Instruction", interactive=True)
                with gr.Row():
                    cfg_img = gr.Number(value=2.5, label="Image Guidance Scale", interactive=True)
                    seed = gr.Number(value=0, label="Random Seed", interactive=True)
                with gr.Row():
                    max_cfg_scale = gr.Slider(value=10.0, minimum=2.0, maximum=20, label="Max Text Guidance Scale", interactive=True)
                    num_steps = gr.Number(value=100, label="Number of Gradual Intervention Steps", interactive=True)
                with gr.Row():
                    run = gr.Button("1. Generate Interventional Data")

            with gr.Column(scale=3, min_width=300):
                with gr.Row():
                    with gr.Column(scale=6, min_width=250):
                        interventions = gr.Image(None, height=150, label="Interventions", interactive=False)
                    with gr.Column(scale=1, min_width=250):
                        interventions_gif = gr.Video(None, height=150, width=150, label="Interventions Animation", interactive=False, loop=True, autoplay=True)
                with gr.Row():
                    xai = gr.Button("2. Explain Trained Classifiers", visible=True)
                with gr.Row():
                    explanation_output = gr.Image(None, visible=True, height=400, label="Model Behavior", interactive=False)
                with gr.Row():
                    null_unb = gr.Image(None, visible=True, height=250, label="Unbiased Null Distribution", interactive=False)
                    null_dcats = gr.Image(None, visible=True, height=250, label="Dark Cats Bias Null Distribution", interactive=False)
                    null_ddogs = gr.Image(None, visible=True, height=250, label="Dark Dogs Bias Null Distribution", interactive=False)

                    

        inputs_run = [input_image, instruction, cfg_img, max_cfg_scale, num_steps, seed]
        outputs_run = [interventions, interventions_gif]
        run.click(fn=explainer.generate_interventions, inputs=inputs_run, outputs=outputs_run)

        inputs_xai = [input_image]
        outputs_xai = [explanation_output, null_unb, null_dcats, null_ddogs]
        xai.click(fn=explainer.explain, inputs=inputs_xai, outputs=outputs_xai)

    demo.launch()