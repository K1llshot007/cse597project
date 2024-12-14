# CSE 597 Project

This repository contains the code and resources for the CSE 597 project. Follow the instructions below to set up the environment, fine-tune the model, generate images, and evaluate the results.

## Prerequisites

1. **Install Dependencies**  
   Ensure that all required dependencies are installed as specified in the `environment` file. Use a virtual environment or conda environment to manage the installations and avoid conflicts.

2. **Update Configuration Paths**  
   The project includes multiple files with hardcoded paths. You must update these paths to match your local directory structure before executing the code.

---

## Model Performance on CUB Dataset (Fine-Tuned Model)

The fine-tuned model achieved the following performance metrics:

- **Inception Score (IS):** 6.53  
- **Frechet Inception Distance (FID):** 30.21  

These metrics demonstrate the effectiveness of the fine-tuning process.

---

## Instructions to Run the Project

### Step 1: Configure the Environment
   Set up the environment as specified in the `environment` file. Use the following steps:
   - Install dependencies using `pip` or `conda` as required.
   - Verify that all libraries and tools are correctly installed.

### Step 2: Download Pre-Trained Diffuser
   Download the pre-trained diffuser model fine-tuned on the extrapolated dataset. Detailed instructions for downloading the model can be found in the original README file included in this repository.  
   - **Optional:** If you prefer, you can train the model from scratch using the provided dataset. Dataset download links are also included in the original README file.

### Step 3: Fine-Tune the Model
   Fine-tune the pre-trained diffuser model using the `train.py` script. Run the following command to begin training:  
   ```bash
   torchrun --nnodes=1 --nproc_per_node=1 --master_port=9902 train.py
   ```
 - **Important Notes:**
     - Many training arguments are set to default values. If specific arguments are not parsed, modify them directly in the script before running the training process.

### Step 4: Generate Images
   After completing the training process, use the `sample_ddp.py` script to generate images from the fine-tuned model.

### Step 5: Evaluate the Model
   #### Inception Score
   - Download the fine-tuned Inception model from this link:  
     [Fine-Tuned Inception Model](https://drive.google.com/file/d/0B3y_msrWZaXLMzNMNWhWdW0zVWs/view?resourcekey=0-gBxxw4fU6ikmNtkfFSQALw).  
   - Use this model to calculate the Inception Score for the generated images. Refer to the documentation in the repository for specific steps.

   #### FID Score
   - Generate `.npz` files for the generated images and the original dataset using the `npz.py` script.  
   - Compute the Frechet Inception Distance (FID) score using these `.npz` files. Detailed instructions can be found in the `npz.py` script.

---

## Additional Notes

- Ensure all datasets, models, and dependencies are correctly downloaded and placed in the appropriate directories.
- Refer to the original README file in the repository for detailed instructions regarding dataset and model downloads.
- Modify the code as needed to suit your specific requirements, such as changing training parameters or updating evaluation scripts.

---
## Original Paper 
**Recurrent Affine Transformation for Text-to-Image Synthesis**  
Senmao Ye, Fei Liu, Minkui Tan  
*arXiv preprint arXiv:2204.10482, 2022*  
[Read the paper on arXiv](https://arxiv.org/abs/2204.10482)
