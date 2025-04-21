import requests
import json

# URL of the search endpoint
url = 'http://localhost:8000/api/search/'

# Search query
data = {
    'query': 'document',
    'fileType': 'all',
    'dateRange': 'all',
    'exactMatch': 'false'
}

# Make the request
response = requests.post(url, data=data)

# Print the response
print(f'Status code: {response.status_code}')
print(f'Response: {json.dumps(response.json(), indent=2)}') 