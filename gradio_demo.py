import gradio as gr
import argparse
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


def get_args():
    parser = argparse.ArgumentParser(description="Interventional Explainer Demo.")
    # we define the start values for the gradio demo here
    parser.add_argument(
        "--example",
        type=str,
        default="example_images/cvd/dog_001.jpg",
        help="Which input example to choose.",
    )
    parser.add_argument(
        "--cfg_img",
        type=float,
        default=2.5,
        help="Image guidance scale for intervention generation.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of gradual intervention steps.",
    )
    parser.add_argument(
        "--max_cfg_scale",
        type=float,
        default=10.0,
        help="Max text guidance scale for intervention generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for intervention generation.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Change the fur color to black. Leave other details the same.",
        help="Intervention instruction.",
    )
    parser.add_argument(
        "--imnet",
        action="store_true",
        default=False,
        help="Use ImageNet model instead of Cats vs Dogs Toy example.",
    )
    return parser.parse_args()


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
    def __init__(self, imnet=False):
        self.imnet = imnet
        # loading the diffusion model
        # we overimplement some of our utilities 
        # so we don't have to load the model multiple times
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
            'timbrooks/instruct-pix2pix', torch_dtype=T.float16, safety_checker=None).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.interventional_data = []
        self.Z = Zollstock()

        self.load_models()

    def load_models(self):
        # we load three models in both the imagenet and cats vs dogs case
        # we also set up the correct transforms and label maps
        if self.imnet:
            print("Loading ImageNet Models...")
            self.M1 = T.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
            self.M1.eval()

            self.M2 = T.hub.load("pytorch/vision", "convnext_tiny", weights="IMAGENET1K_V1")
            self.M2.eval()

            self.M3 = T.hub.load("pytorch/vision", "vit_b_16", weights="IMAGENET1K_V1")
            self.M3.eval()
            
            self.test_transform = transforms.Compose([
                # Resize the images to IMAGE_SIZE xIMAGE_SIZE
                transforms.Resize(size=(224, 224)),
                # Turn the image into a torch.Tensor
                # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
                transforms.ToTensor(),
                # normalize to [-1,1]
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            with open("imnet_classes.txt", "r") as f:
                labels = [x.replace("_", " ").title() for x in f.read().splitlines()]
            self.label_map = labels

            self.logit_of_interest = (281, 281, 281)  # tabby cat class

        else:
            print("Loading Cats vs Dogs Models...") 
            # loading models for later explanation
            print("Unbiased:")
            self.M1, _ = load_cm("checkpoints/cats-vs-dogs/convmixer_unbiased.pth", verbose=True)
            print("\nCats have dark fur bias:")
            self.M2, _ = load_cm("checkpoints/cats-vs-dogs/convmixer_dark-cats-bias.pth", verbose=True)
            print("\nDogs have dark fur bias:")
            self.M3, _ = load_cm("checkpoints/cats-vs-dogs/convmixer_dark-dogs-bias.pth", verbose=True)

            self.test_transform = transforms.Compose([
                # Resize the images to IMAGE_SIZE xIMAGE_SIZE
                transforms.Resize(size=(128, 128)),
                # Turn the image into a torch.Tensor
                # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
                transforms.ToTensor(),
                # normalize to [-1,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.logit_of_interest = (0, 0, 0)  # cat class
            self.label_map = ["Cat", "Dog"]



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
        logits_M1 = perform_inference(self.M1, loader)
        logits_M2 = perform_inference(self.M2, loader)
        logits_M3 = perform_inference(self.M3, loader)

        if self.imnet:
            self.logit_of_interest = (np.argmax(logits_M1[0]), np.argmax(logits_M2[0]), np.argmax(logits_M3[0]))


        propgrad_M1, pval_M1, null_dist_M1 = self.Z.shuffle_test(logits_M1[:,self.logit_of_interest[0]], plot_null_dist=True)
        null_dist_M1.savefig("tmp/null_M1.png", transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(null_dist_M1)
        
        propgrad_M2, pval_M2, null_dist_M2 = self.Z.shuffle_test(logits_M2[:,self.logit_of_interest[1]], plot_null_dist=True)
        null_dist_M2.savefig("tmp/null_M2.png", transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(null_dist_M2)

        propgrad_M3, pval_M3, null_dist_M3 = self.Z.shuffle_test(logits_M3[:,self.logit_of_interest[2]], plot_null_dist=True)
        null_dist_M3.savefig("tmp/null_M3.png", transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(null_dist_M3)

        print("M1 - Prediction on Original Image:", 
              np.argmax(logits_M1[0]), 
              logits_M1[0, np.argmax(logits_M1[0])],
              f"({self.label_map[np.argmax(logits_M1[0])]})")
        print("M1 - Prediction on Full Intervention:", 
              np.argmax(logits_M1[-1]), 
              logits_M1[-1, np.argmax(logits_M1[-1])],
              f"({self.label_map[np.argmax(logits_M1[-1])]})")
        print(f"M1 - PropGrad: {propgrad_M1:.4f}, p-value: {pval_M1:.4f}\n")
        
        print("M2 - Prediction on Original Image:", 
              np.argmax(logits_M2[0]), 
              logits_M2[0, np.argmax(logits_M2[0])],
              f"({self.label_map[np.argmax(logits_M2[0])]})")
        print("M2 - Prediction on Full Intervention:", 
              np.argmax(logits_M2[-1]), 
              logits_M2[-1, np.argmax(logits_M2[-1])],
              f"({self.label_map[np.argmax(logits_M2[-1])]})")
        print(f"M2 - PropGrad: {propgrad_M2:.4f}, p-value: {pval_M2:.4f}\n")
        
        print("M3 - Prediction on Original Image:", 
              np.argmax(logits_M3[0]), 
              logits_M3[0, np.argmax(logits_M3[0])],
              f"({self.label_map[np.argmax(logits_M3[0])]})")
        print("M3 - Prediction on Full Intervention:", 
              np.argmax(logits_M3[-1]), 
              logits_M3[-1, np.argmax(logits_M3[-1])],
              f"({self.label_map[np.argmax(logits_M3[-1])]})")        
        print(f"M3 -  PropGrad: {propgrad_M3:.4f}, p-value: {pval_M3:.4f}\n")
        
        fig = plt.figure(figsize=(14,5))
        ax = fig.add_subplot(111)

        # order should be cats, dogs
        me = max(1, (len(self.interventional_data)+1) // 7)
        lw = 3
        ms = 12

        c_M1 = "#f72585" 
        c_M2 = "#3a0ca3" 
        c_M3 = "#4895ef" 

        num = len(self.interventional_data) +1
        vals = np.arange(num)

        if self.imnet:
            lengths = [len(self.label_map[x]) for x in self.logit_of_interest]
            if max(lengths) > 7:
                inbetween = "\n"
            else:
                inbetween = " "
                
            model_labels = [f"ResNet-50{inbetween}({self.label_map[self.logit_of_interest[0]]})", 
                            f"ConvNext-T{inbetween}({self.label_map[self.logit_of_interest[1]]})",
                            f"ViT-B/16{inbetween}({self.label_map[self.logit_of_interest[2]]})"]
        else:
            model_labels = ["Unbiased", "Dark Cats Bias", "Dark Dogs Bias"]
        ax.plot(vals, logits_M1[:,self.logit_of_interest[0]], label=f"{model_labels[0]}\n" + r"$\mathbb{E}[|\nabla_\mathsf{X}\mathbb{F}|] = " + f"{propgrad_M1:.3f}$\n" + f"$p$-val = {pval_M1:.3f}", color=c_M1, linestyle="-", linewidth=lw, )
        ax.plot(vals, logits_M2[:,self.logit_of_interest[1]], label=f"{model_labels[1]}\n" + r"$\mathbb{E}[|\nabla_\mathsf{X}\mathbb{F}|] = " + f"{propgrad_M2:.3f}$\n" + f"$p$-val = {pval_M2:.3f}", color=c_M2, linestyle=":", linewidth=lw, marker="*", markevery=me, markersize=ms)
        ax.plot(vals, logits_M3[:,self.logit_of_interest[2]], label=f"{model_labels[2]}\n" + r"$\mathbb{E}[|\nabla_\mathsf{X}\mathbb{F}|] = " + f"{propgrad_M3:.3f}$\n" + f"$p$-val = {pval_M3:.3f}", color=c_M3, linestyle=":", linewidth=lw, marker="v", markevery=me, markersize=ms)
        
        if not self.imnet:
            ax.axhline(0.5, color="r", linestyle="--")
            ax.set_ylim((-0.1,1.1)),20
            ax.set_yticks([0,1], ["Dog", "Cat"])
            ax.set_ylabel("Prediction")
        else:
            max_val = max(np.max(logits_M1[:,self.logit_of_interest[0]]), np.max(logits_M2[:,self.logit_of_interest[1]]), np.max(logits_M3[:,self.logit_of_interest[2]]))
            ax.set_ylim((-0.1, max_val*1.1))
            ax.set_ylabel("Prediction Logit Value") 


        ax.set_xticks([num*0.05,(num-1)*0.95], ["Original", "Interventional"])
        ax.set_xlabel("Data")

        legend_title = "ConvMixer Models" if not self.imnet else "Models (Pred. Class)"
        ax.legend(loc='center right', bbox_to_anchor=(1.55, 0.5), title=legend_title,
                ncol=1, fancybox=True, shadow=True) 

        fig.tight_layout(pad=0.2)
        if not Path("tmp").exists():
            Path("tmp").mkdir(parents=True, exist_ok=True)

        fig.savefig("tmp/explanation.png", transparent=False, bbox_inches='tight', pad_inches=0.2)
        plt.close('all')

        return gr.update(value="tmp/explanation.png", visible=True), gr.update(value="tmp/null_M1.png", visible=True), gr.update(value="tmp/null_M2.png", visible=True), gr.update(value="tmp/null_M3.png", visible=True)
    


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
    args = get_args()
    explainer = Interventional_Explainer(imnet=args.imnet)

    # just for displaying correct labels for the toy example and the imagenet case
    if args.imnet:
        null_dist_labels = ["ResNet-50 Null Distribution", "ConvNext-T Null Distribution", "ViT-B/16 Null Distribution"]
    else:
        null_dist_labels = ["Unbiased Null Distribution", "Dark Cats Bias Null Distribution", "Dark Dogs Bias Null Distribution"]

    demo = gr.Blocks()
    with demo:
        with gr.Row():   
            with gr.Column(scale=2, min_width=300):
                with gr.Row():
                    input_image = gr.Image(args.example, height=300, width=400, label="Input Image", interactive=True)
                with gr.Row():
                    instruction = gr.Textbox(value=args.instruction, max_lines=3, label="Intervention Instruction", interactive=True)
                with gr.Row():
                    cfg_img = gr.Number(value=args.cfg_img, label="Image Guidance Scale", interactive=True)
                    seed = gr.Number(value=args.seed, label="Random Seed", interactive=True)
                with gr.Row():
                    max_cfg_scale = gr.Slider(value=args.max_cfg_scale, minimum=2.0, maximum=20, label="Max Text Guidance Scale", interactive=True)
                    num_steps = gr.Number(value=args.num_steps, label="Number of Gradual Intervention Steps", interactive=True)
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
                    null_M1 = gr.Image(None, visible=True, height=250, label=null_dist_labels[0], interactive=False)
                    null_M2 = gr.Image(None, visible=True, height=250, label=null_dist_labels[1], interactive=False)
                    null_M3 = gr.Image(None, visible=True, height=250, label=null_dist_labels[2], interactive=False)

                    

        inputs_run = [input_image, instruction, cfg_img, max_cfg_scale, num_steps, seed]
        outputs_run = [interventions, interventions_gif]
        run.click(fn=explainer.generate_interventions, inputs=inputs_run, outputs=outputs_run)

        inputs_xai = [input_image]
        outputs_xai = [explanation_output, null_M1, null_M2, null_M3]
        xai.click(fn=explainer.explain, inputs=inputs_xai, outputs=outputs_xai)

    demo.launch()