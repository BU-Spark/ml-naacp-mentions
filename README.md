# Deployment instructions

## File structure:

Deployment/
\|--|
\|  |--AICodeInit/
\|     |--\__init__.py
\|     |--LlamaCode.py
\|     |--spacy_textblob_functions.py
\|
\|--app.py
\|--requirements.txt

* **app.py**: Includes the main logic for Huggingface application
* **requirements.txt**: All the dependency required for the application (Llama, SpaCy, etc.)
* **AICodeInit/**: helper functions for ML pipeline

## Steps to deploy the code:

### 1. Create Huggingface Space (Assuming you have a huggingface account)

**To make a new Space**, visit the [Spaces main page](https://huggingface.co/spaces) and click on **Create new Space**. Along with choosing a name for your Space, selecting an optional license, and setting your Space’s visibility, you’ll be prompted to choose the **SDK** for your Space. The Hub offers four SDK options: Gradio, Streamlit, Docker and static HTML. If you select “Gradio” as your SDK, you’ll be navigated to a new repo showing the following page:

![img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/spaces-blank-space.png)

Under the hood, Spaces stores your code inside a git repository, just like the model and dataset repositories. Thanks to this, the same tools we use for all the [other repositories on the Hub](https://huggingface.co/docs/hub/en/repositories) (`git` and `git-lfs`) also work for Spaces. Follow the same flow as in [Getting Started with Repositories](https://huggingface.co/docs/hub/en/repositories-getting-started) to add files to your Space. Each time a new commit is pushed, the Space will automatically rebuild and restart.

For step-by-step tutorials to creating your first Space, see the guides below:

- [Creating a Gradio Space](https://huggingface.co/docs/hub/en/spaces-sdks-gradio)
- [Creating a Streamlit Space](https://huggingface.co/docs/hub/en/spaces-sdks-streamlit)
- [Creating a Docker Space](https://huggingface.co/docs/hub/en/spaces-sdks-docker-first-demo)

### 2. Get Permission to use Llama model

To download and use the Llama 2 model, simply fill out Meta’s form to request access. Please note that utilizing Llama 2 is contingent upon accepting the Meta license agreement. 

**Important: Use your email linked with your huggingface account to submit the permission request**

https://llama.meta.com/llama-downloads/?source=post_page-----3a29fdbaa9ed--------------------------------

### 3. Create Access token on huggingface account

Click your avatar on the upper right corner and go to your huggingface account settings. Click "access token" page and generate an access token. 

Then, put the name of the access token as an argument to `os.getenv()` method.

![image-20240507125121115](/Users/jialuli/Library/Application Support/typora-user-images/image-20240507125121115.png)

### 4. Run the application

Click the `App` tab in your huggingface space. It will auto-deploy and provide access to the model via Gradio interface.
