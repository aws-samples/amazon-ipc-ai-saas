pip3 install --target ./package -r requirements.txt
cd package
zip -r ../face_query.zip .
cd ../
zip -g face_query.zip main.py