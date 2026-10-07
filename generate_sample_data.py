"""Generate a synthetic social media posts dataset (data/posts.csv)."""
import random
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)

TOPICS = {
    "#AI": ["AI is changing how we work", "Excited about the new AI models", "AI hype is getting out of hand"],
    "#MachineLearning": ["Just finished a great ML course", "Struggling with model overfitting again", "Love how clean this ML pipeline turned out"],
    "#Python": ["Python makes data work so easy", "Debugging Python all night, awful", "Best language for quick prototypes"],
    "#DataScience": ["Data science roles are booming", "Messy data ruins my day", "Beautiful dashboard shipped today"],
    "#Cricket": ["What a fantastic match today", "Terrible umpiring decision", "Our team played brilliantly"],
    "#Startup": ["Launch day, feeling great", "Funding is tough right now", "Proud of our small team"],
    "#Travel": ["Amazing sunset in Goa", "Flight delayed again, so annoying", "Best trip of the year"],
    "#Food": ["This biryani is incredible", "Overpriced and bland, disappointing", "Street food never disappoints"],
    "#Tech": ["New phone launch looks impressive", "Battery life is horrible", "Great update, works smoothly"],
    "#Fitness": ["Morning run done, feeling strong", "Skipped the gym, feeling lazy", "Great workout today"],
}
WEIGHTS = [10, 9, 8, 7, 6, 5, 4, 4, 3, 2]
tags = list(TOPICS)
start = datetime(2026, 9, 1)
rows = []
for i in range(500):
    main = random.choices(tags, WEIGHTS)[0]
    extra = random.sample(tags, k=random.choice([0, 1, 1, 2]))
    text = random.choice(TOPICS[main])
    all_tags = [main] + [t for t in extra if t != main]
    rows.append({
        "id": i + 1,
        "date": (start + timedelta(days=random.randint(0, 29), hours=random.randint(0, 23))).strftime("%Y-%m-%d %H:%M"),
        "text": f"{text} {' '.join(all_tags)}",
    })
pd.DataFrame(rows).to_csv("data/posts.csv", index=False)
print("Saved data/posts.csv with", len(rows), "posts")
