from flask import Flask, render_template, request, send_file, session
import joblib
import numpy as np
from scraper import scrape_instagram
from reportlab.pdfgen import canvas
import requests
from datetime import datetime
import numpy as np # Note: np imported twice in original, but leaving for safety

# ✅ Working Twitter Data Fetcher
def get_twitter_features(username, bearer_token):
    try:
        url = f"https://api.twitter.com/2/users/by/username/{username}?user.fields=public_metrics,description,profile_image_url,verified,created_at"
        headers = {"Authorization": f"Bearer {bearer_token}"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print("❌ Twitter API error:", response.status_code, response.text)
            return None

        data = response.json()
        user = data.get("data", {})
        metrics = user.get("public_metrics", {})

        followers = metrics.get("followers_count", 0)
        following = metrics.get("following_count", 0)
        posts = metrics.get("tweet_count", 0)
        bio = 1 if user.get("description") else 0
        profile_pic = 1 if user.get("profile_image_url") else 0
        verified = 1 if user.get("verified") else 0

        # Account age
        account_created = user.get("created_at")
        if account_created:
            created_date = datetime.strptime(account_created, "%Y-%m-%dT%H:%M:%S.%fZ")
            account_age_days = (datetime.utcnow() - created_date).days
        else:
            account_age_days = 0

        # Fake engagement (demo purpose)
        engagement = round(np.random.uniform(0.1, 0.8), 2)

        # ✅ Return as dictionary (very important)
        return {
            "followers": followers,
            "following": following,
            "posts": posts,
            "bio": bio,
            "profile_pic": profile_pic,
            "verified": verified,
            "account_age_days": account_age_days,
            "engagement": engagement
        }

    except Exception as e:
        print("⚠️ Twitter fetch error:", e)
        return None


# ===========================
# FEATURE LISTS FOR EACH PLATFORM
# ===========================
INSTA_FEATURES = ["followers", "following", "posts", "bio", "profile_pic", "engagement"]
TW_FEATURES = ["followers", "following", "posts", "bio", "profile_pic", "verified", "account_age_days", "engagement"]
FB_FEATURES = ["followers", "following", "posts", "bio", "profile_pic"]

# ===========================
# FLASK APP CONFIG
# ===========================
app = Flask(__name__)
app.secret_key = "315e3414edb6b308142ab7a326886142"

# ===========================
# LOAD TRAINED MODELS
# ===========================
rf_insta = joblib.load("insta_model.pkl")
rf_tw = joblib.load("twitter_model.pkl")
rf_fb = joblib.load("facebook_model.pkl")

print("✅ Models loaded successfully!")

# ===========================
# EXPLANATION GENERATOR
# ===========================
def explain_prediction(model, x_array, FEATURES):
    try:
        model_core = model.named_steps[model.steps[-1][0]]
    except AttributeError:
        model_core = model

    if hasattr(model_core, "feature_importances_"):
        importances = model_core.feature_importances_
        pairs = sorted(zip(FEATURES, importances), key=lambda x: -x[1])[:3]
        return "Top contributing features: " + ", ".join(
            [f"{f} ({round(v, 3)})" for f, v in pairs]
        )

    elif hasattr(model_core, "coef_"):
        coef = model_core.coef_[0]
        pairs = sorted(zip(FEATURES, coef), key=lambda x: -abs(x[1]))[:3]
        return "Top contributing features: " + ", ".join(
            [f"{f} ({round(v, 3)})" for f, v in pairs]
        )

    else:
        return "No model explanation available."

# ===========================
# CHAT EXPLANATION
# ===========================
def generate_chat_explanation(values, verdict, score, FEATURES):
    try:
        bio = values[FEATURES.index("bio")] if "bio" in FEATURES else 0
        profile_pic = values[FEATURES.index("profile_pic")] if "profile_pic" in FEATURES else 0
        followers = values[FEATURES.index("followers")] if "followers" in FEATURES else 0
        following = values[FEATURES.index("following")] if "following" in FEATURES else 0
        posts = values[FEATURES.index("posts")] if "posts" in FEATURES else 0
    except:
        return "Unable to generate detailed chat explanation."

    if verdict == "Fake":
        if bio == 0 and profile_pic == 0 and followers < 50:
            return ("This account lacks a bio and profile picture, and has very few followers. "
                    "These are classic signs of a fake or inactive profile.")
        elif posts < 5 and followers < 100:
            return ("Extremely low posting activity and follower count suggest this account may be automated or abandoned.")
        elif followers > 1000 and posts < 5:
            return ("High followers but very low posts — this could indicate purchased followers or deceptive activity.")
        elif bio == 0:
            return ("Missing bio raises concerns about authenticity.")
        else:
            return ("Several profile completeness and interaction factors indicate suspicious behavior, "
                    f"leading to a high risk score of {score}%.")
    else:
        if bio == 1 and profile_pic == 1:
            return ("This account has a complete profile, which indicates authenticity.")
        elif followers > 500 and posts > 10:
            return ("With a healthy follower base and consistent posting, this account appears genuine.")
        elif posts >= 5:
            return ("Moderate posting activity suggests this account is likely real.")
        else:
            return ("While not very active, the account shows signs of legitimacy.")

# ===========================
# PDF GENERATOR
# ===========================
def generate_pdf(data, filename="report.pdf"):
    c = canvas.Canvas(filename)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Fake Account Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 770, f"Username: {data['username']}")
    c.drawString(100, 750, f"Verdict: {data['verdict']}")
    c.drawString(100, 730, f"Risk Score: {data['score']}%")
    c.drawString(100, 710, f"Explanation: {data['explanation']}")

    y = 690
    c.drawString(100, y, "Input Features:")
    for feature, value in data['values']:
        y -= 20
        c.drawString(120, y, f"{feature}: {value}")

    c.save()

# ===========================
# ROUTES
# ===========================
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/instagram')
def instagram():
    return render_template('instagram.html')

@app.route('/facebook')
def facebook():
    return render_template('facebook.html')

@app.route('/twitter')
def twitter():
    return render_template('twitter.html')


# =========================================================
# ✅ UNIFIED ML PREDICTION ROUTE (FIXES 404 ERROR)
# =========================================================
@app.route('/predict', methods=['POST']) 
def predict_account():
    mode = request.form.get("mode")

    # ---------- INSTAGRAM ----------
    if mode == "public":
        username = request.form.get("username")
        scraped = scrape_instagram(username)
        if not isinstance(scraped, dict) or "error" in scraped:
            return render_template("result.html", verdict="Error", score=0,
                                   explanation=scraped.get("error", "Unknown error"), values=[])
        values = [scraped.get(k, 0) for k in INSTA_FEATURES]
        profile_pic_url = scraped.get("profile_pic_url")
        model = rf_insta
        FEATURES = INSTA_FEATURES
        
    elif mode == "private":
        username = request.form.get("username", "Manual Instagram Entry")
        followers = float(request.form.get("followers", 0))
        following = float(request.form.get("following", 0))
        posts = float(request.form.get("posts", 0))
        bio = float(request.form.get("bio", 0))
        profile_pic = float(request.form.get("profile_pic", 0))
        engagement = 0.0 #Dummy for consistency

        values = [followers, following, posts, bio, profile_pic, engagement]
        model = rf_insta
        FEATURES = ["followers", "following", "posts", "bio", "profile_pic", "engagement"]
        profile_pic_url = None

    # ---------- TWITTER ----------
    elif mode == "twitter":
        username = request.form.get("username")
        bearer_token = "AAAAAAAAAAAAAAAAAAAAAHOW4wEAAAAAk3rzIwC96JxWZkD0%2FMxHpE5tWkE%3DGQGyHaVHnDYIEZo61ZNOYRzGj8PO69nL9HN5asHDHTDtUxbasX"
        
        # Get data from scraper (must return a dict)
        twitter_data = get_twitter_features(username, bearer_token)
        
        if not twitter_data or not isinstance(twitter_data, dict):
            return render_template("result.html", username=username, verdict="Error", score=0,
                                   explanation="Failed to fetch Twitter data.", values=[], chat_explanation=None)
        
        # Safely extract all 8 features
        followers = twitter_data.get("followers", 0)
        following = twitter_data.get("following", 0)
        posts = twitter_data.get("posts", 0)
        bio = twitter_data.get("bio", 0)
        profile_pic = twitter_data.get("profile_pic", 0)
        verified = twitter_data.get("verified", 0)
        account_age_days = twitter_data.get("account_age_days", 0)
        engagement = twitter_data.get("engagement", 0)

        # Ensure correct order
        values = [followers, following, posts, bio, profile_pic, verified, account_age_days, engagement]
        
        model = rf_tw
        FEATURES = TW_FEATURES
        profile_pic_url = None


    # ---------- FACEBOOK ----------
    elif mode in ["facebook", "facebook_private"]:
        username = request.form.get("username", "Manual Facebook Entry")
        followers = float(request.form.get("followers", 0))
        following = float(request.form.get("following", 0))
        posts = float(request.form.get("posts", 0))
        bio = float(request.form.get("bio", 0))
        profile_pic = float(request.form.get("profile_pic", 0))

        values = [followers, following, posts, bio, profile_pic]
        model = rf_fb
        FEATURES = FB_FEATURES
        profile_pic_url = None

    else:
        return "Invalid mode", 400

    # ---------- MODEL PREDICTION ----------
    x = np.array(values).reshape(1, -1)
    
    # Check if model supports predict_proba (most do, but defensive coding)
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(x)[0][1]
    else:
        # Fallback if model only supports binary prediction
        prob = model.predict(x)[0]
    
    risk_score = round(prob * 100, 2)
    verdict = "Fake" if risk_score >= 50 else "Real"

    explanation = explain_prediction(model, x, FEATURES)
    chat_explanation = generate_chat_explanation(values, verdict, risk_score, FEATURES)
    feature_value_pairs = list(zip(FEATURES, values))

    session.update({
        "username": username,
        "verdict": verdict,
        "score": risk_score,
        "explanation": explanation,
        "values": feature_value_pairs,
        "chat_explanation": chat_explanation,
        "profile_pic_url": profile_pic_url
    })

    return render_template("result.html",
                           username=username,
                           verdict=verdict,
                           score=risk_score,
                           explanation=explanation,
                           values=feature_value_pairs,
                           chat_explanation=chat_explanation,
                           profile_pic_url=profile_pic_url)

# ---------- PDF DOWNLOAD ----------
@app.route("/download-report")
def download_report():
    data = {
        "username": session.get("username", "N/A"),
        "verdict": session.get("verdict", "N/A"),
        "score": session.get("score", 0),
        "explanation": session.get("explanation", "N/A"),
        "values": session.get("values", [])
    }
    generate_pdf(data)
    return send_file("report.pdf", as_attachment=True)

# ---------- REPORT ROUTE ----------
@app.route("/report", methods=["POST"])
def report_account():
    username = request.form.get("username")
    return render_template("result.html",
                           verdict=session.get("verdict"),
                           score=session.get("score"),
                           explanation=session.get("explanation"),
                           values=session.get("values"),
                           chat_explanation=session.get("chat_explanation"),
                           profile_pic_url=session.get("profile_pic_url"),
                           message=f"✅ Account '{username}' has been reported successfully.")

# ===========================
# RUN APP
# ===========================
if __name__ == "__main__":
    app.run(debug=True)