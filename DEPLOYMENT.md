# Run and deploy Telco_Customer_Churn

## 1. Run locally
1. Extract the project ZIP.
2. Open the project folder in VS Code or a terminal.
3. Create and activate a virtual environment.
4. Install dependencies with `pip install -r requirements.txt`.
5. Start the app with `streamlit run app.py`.
6. The app loads `Telco_Customer_Churn.csv` automatically because it is included beside `app.py`.

## 2. Upload to GitHub
Create a new GitHub repository, for example `telco-customer-churn`.

From the project folder, run:

```bash
git init
git add .
git commit -m "Add Telco customer churn prediction project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/telco-customer-churn.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Authenticate with GitHub when Git prompts you. Do not commit passwords or access tokens.

## 3. Deploy on Streamlit Community Cloud
1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Select your repository and `main` branch.
4. Set the main file path to `app.py`.
5. Deploy and wait for the build to finish.
6. Open the generated public app URL and test the dataset insights and prediction form.

## 4. Dataset privacy and licensing
This repository currently includes the CSV uploaded for this project. Before making the repository public, confirm that the dataset license permits redistribution. If not, remove the CSV from Git tracking and configure the app to accept an uploaded copy instead.
