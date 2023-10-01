import pandas as pd
from PIL import Image, ImageStat

# Variation in contrast across types
image_stats = pd.DataFrame()

def load_im_get_stats(file, path, tumor_class, tumor_type): 
    results = pd.DataFrame()
    
    im = Image.open(path)
    stats = ImageStat.Stat(im)
    
    # get stats for image
    for band,channel in enumerate(im.getbands()): 
        # print(f'Band: {name}, min/max: {stats.extrema[band]}, stddesv: {stats.stddev[band]}')
        temp = {
            'TUMOR_CLASS':tumor_class,
            'TUMOR_TYPE' : tumor_type,
            'channel': channel,
            'Min' : [stats.extrema[band][0]],
            'Max' : [stats.extrema[band][1]],
            'stddev' : [stats.stddev[band]]
        }

        results = pd.concat([results, pd.DataFrame.from_dict(temp)])
        
        del temp
    
    return results
    
    
color_variation = pd.DataFrame()

for i in range(len(df)): 
    file, path, tumor_class, tumor_type = df.loc[i, ['BASE_NAME', 'FULL_PATH', 'TUMOR_CLASS', 'TUMOR_TYPE']]
    temp = load_im_get_stats(file, path, tumor_class, tumor_type)
    color_variation = pd.concat([color_variation, pd.DataFrame.from_dict(temp)])
    del temp
    
color_variation.to_csv(os.path.join(r"C:\Users\nickl\Documents\Berkeley\W281\Computer-Vision-281-Final-Project\data", "color_variation.csv"))