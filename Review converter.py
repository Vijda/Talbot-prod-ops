import pandas as pd
from deep_translator import GoogleTranslator

# Load your Excel file
df = pd.read_excel("/Users/vdevados/Documents/Python Scripts/Talabat/Talabat Reviews.xlsx")

# Translate reviews
df['Translated Review'] = df['Review'].apply(lambda text: GoogleTranslator(source='auto', target='en').translate(text))

# Save to new Excel
df.to_excel("Talabat_Reviews_Translated.xlsx", index=False)
