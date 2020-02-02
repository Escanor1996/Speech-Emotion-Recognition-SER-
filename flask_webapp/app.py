import os
from flask import Flask, render_template,request
app = Flask(__name__)

from inference import prediction

@app.route('/',methods=['GET','POST'])
def hello_world():
	if request.method=='GET':
		return render_template('index.html')
	if request.method=='POST':
		try:
			file=request.files['file']
			image=file.read()
			emotion=prediction(image_bytes=image)
			return render_template('result.html',emotions=emotion)

			
		except:

			return render_template('index.html')
		

    

if __name__ == '__main__':
	app.run(debug=True,port=os.getenv('PORT',5000))
	




				
		    
		    
		    