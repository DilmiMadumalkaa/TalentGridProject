# from google_auth_oauthlib.flow import InstalledAppFlow

# # Define the Gmail API scope
# SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# def main():
#     # Authenticate using credentials.json
#     flow = InstalledAppFlow.from_client_secrets_file(
#         'credentials.json', SCOPES)
    
#     # Launch a browser for authentication and generate the token
#     creds = flow.run_local_server(port=0)
    
#     # Save the token to token.json
#     with open('token.json', 'w') as token_file:
#         token_file.write(creds.to_json())
#     print("Token has been generated and saved as token.json")

# if __name__ == '__main__':
#     main()


import requests

# Define the token endpoint and your parameters
token_url = "https://oauth2.googleapis.com/token"
data = {
    'code': '4%2F0AanRRrujLd0lHLzCczaSzAMZEKX4jPEgUkns0aZUxCtPQHUqr7gWB5WLiSohKwQHOveyzw&redirect_uri=https%3A%2F%2Fdevelopers.google.com%2Foauthplayground&client_id=514464613650-95jc54bdesgoiha8kf852bebo4jpbja1.apps.googleusercontent.com&client_secret=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&scope=&grant_type=authorization_code',  
    'client_id': '514464613650-95jc54bdesgoiha8kf852bebo4jpbja1.apps.googleusercontent.com',
    'client_secret': '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',
    'redirect_uri': 'https://developers.google.com/oauthplayground',  # Same as the redirect URI you used
    'grant_type': 'authorization_code'
}

# Send the POST request to exchange the code for a token
response = requests.post(token_url, data=data)

# Check the response
if response.status_code == 200:
    print(response.json())  # This should contain the access token and refresh token
else:
    print(f"Error: {response.status_code}")
    print(response.json())  # This will give more details on what went wrong
