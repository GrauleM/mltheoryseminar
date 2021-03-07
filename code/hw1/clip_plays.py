def get_text_features(text):
    with torch.no_grad():
        # Encode and normalize the description using CLIP
        text_encoded = model.encode_text(clip.tokenize(text).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    text_features = (text_encoded.cpu().numpy())

    return text_features


def get_closest_photo_feature(text, photo_features):
    text_features = get_text_features(text)
    similarities = list((text_features @ photo_features.T).squeeze(0))
    # Retrieve the photo ID of the best photo
    idx = best_photos[0][1]
    # get the photo features for the best photo
    photo_features = photo_features[idx]

    return photo_features


def retrieve_similar_photos(query_features, n_best):
    similarities = list((query_features @ photo_features.T).squeeze(0))

    # Sort the photos by their similarity score
    best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)
    best_photo_ids = []
    # Iterate over the top 3 results
    for i in range(n_best):
        # Retrieve the photo ID
        idx = best_photos[i][1]
        photo_id = photo_ids[idx]
        best_photo_ids.append(photo_id)
    return best_photo_ids


def show_photos(photo_ids):
    for photo_id in photo_ids:
        # Get all metadata for this photo
        photo_data = photos[photos["photo_id"] == photo_id].iloc[0]

        # Display the photo
        display(Image(url=photo_data["photo_image_url"] + "?w=640"))

        # Display the attribution text
        display(HTML(
            f'Photo by <a href="https://unsplash.com/@{photo_data["photographer_username"]}?utm_source=NaturalLanguageImageSearch&utm_medium=referral">{photo_data["photographer_first_name"]} {photo_data["photographer_last_name"]}</a> on <a href="https://unsplash.com/?utm_source=NaturalLanguageImageSearch&utm_medium=referral">Unsplash</a>'))
        print()


coeffs = [1, 3, 3]
query_features = 1 / (coeffs[0] + coeffs[1] + coeffs[2]) * (
            coeffs[0] * get_closest_photo_feature("black dog", photo_features) - coeffs[1] * get_text_features(
        "black") + coeffs[2] * get_text_features("white"))

best_photo_ids = retrieve_similar_photos(query_features, 1)

show_photos(best_photo_ids)