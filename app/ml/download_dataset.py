import os
import gdown

if not os.path.isdir(os.path.join(os.path.dirname(__file__), "dataset")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "dataset"))

gdown.download(id='1bL9Q7U5-Hki054FwcDzTmgpuf1LOv5Su', output=os.path.join(os.path.dirname(__file__), './dataset/train.csv'))
gdown.download(id='1dgclPFmCHatMIGQig9UHrpbYRi2v-Ne-', output=os.path.join(os.path.dirname(__file__),'./dataset/holiday_data.csv'))
gdown.download(id='1B8hMmhhBx7UOtd-dF1dmY-ZXgNe2_ixk', output=os.path.join(os.path.dirname(__file__),'./dataset/osrm_data_train.csv'))
gdown.download(id='152UC7VWgEdMr-5qAwTKMf66ja2_0IRMT', output=os.path.join(os.path.dirname(__file__),'./dataset/weather_data.csv'))
