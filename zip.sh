mkdir package

cp -r ./IMG/ package/IMG/
cp ./lambda_function.py ./package/

pip install --target ./package -r requirements.txt

cd package
zip package.zip -r .

cd ..

mv ./package/package.zip .
rm -r ./package

