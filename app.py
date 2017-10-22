from bottle import hook, request, response, route, run
from sklearn.externals import joblib

mlb, classifier = joblib.load('offClassifier.pkl')


@hook('after_request')
def enable_cors():
    """
    You need to add some headers to each request.
    Don't use the wildcard '*' for Access-Control-Allow-Origin in production.
    """
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'


@route('/predict', method=['OPTIONS', 'POST'])
def predict():
    if request.method == 'OPTIONS':
        return {}

    products = request.json
    predictions = mlb.inverse_transform(
        classifier.predict([p['name'] for p in products])
    )
    return {
        'data': [
            product.update({'predictedCategories': categories}) or product
            for product, categories in zip(products, predictions)
        ]
    }


if __name__ == '__main__':
    run(host='localhost', port=4242)
