from dotenv import load_dotenv
import os
import telebot
import urllib.request
from os import listdir
import cv2
import numpy as np

load_dotenv()

# SIFT-HISTOGRAM

sift = cv2.SIFT_create()
# feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

MIN_RES = 300
DIST_RATIO = 0.6
MATCHES_THRESHOLD = 8

bot = telebot.TeleBot(os.environ.get('TELEGRAM_BOT_API_KEY'))

folder = 'pics'

hists = {}
keypoints = {}
descriptors = {}
states = {} # 0 - upload, 1 - search


def get_hist(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def get_descriptor_from_file(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_height = np.size(image, 0)
    img_width = np.size(image, 1)

    downscale = 1
    if img_height > MIN_RES:
        downscale = int(img_height / MIN_RES)
    elif img_width > MIN_RES:
        downscale = int(img_width / MIN_RES)

    dimensions = (int(img_width / downscale), int(img_height / downscale))

    image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return get_descriptor(image)


def get_descriptor(image):
    return sift.detectAndCompute(image, None)


def get_good_points(descriptor_a, descriptor_b):
    matches = flann.knnMatch(descriptor_a, descriptor_b, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < DIST_RATIO * n.distance:
            good_points.append(m)
    return len(good_points)


def are_same_descriptors(descriptor_a, descriptor_b):
    return get_good_points(descriptor_a, descriptor_b) > MATCHES_THRESHOLD


def get_file_download_url(file_path):
    return 'https://api.telegram.org/file/bot'+os.environ.get('TELEGRAM_BOT_API_KEY')+'/'+file_path


def is_file_in_folder(file_path):

    pics = listdir(folder)

    keypoints[file_path], descriptors[file_path] = get_descriptor_from_file(file_path)
    hists[file_path] = get_hist(file_path)

    for db_pic in pics:

        if db_pic not in descriptors:
            keypoints[db_pic], descriptors[db_pic] = get_descriptor_from_file(folder + '/' + db_pic)
        if db_pic not in hists:
            hists[db_pic] = get_hist(folder + '/' + db_pic)

        good_points_number = '-'
        hist_compare = cv2.compareHist(hists[db_pic], hists[file_path], cv2.HISTCMP_BHATTACHARYYA)

        if hist_compare > 0.8:
            found_same = False
        else:
            good_points_number = get_good_points(descriptors[db_pic], descriptors[file_path])
            found_same = good_points_number > MATCHES_THRESHOLD

        print('compare', db_pic, 'and', file_path)
        print('hist_compare:', hist_compare)
        print('good_points_number:', good_points_number)

        if found_same:
            return found_same, db_pic

    return False, ''


def set_state(message, state):
    uuid = message.from_user.id
    if state in [0, 1]:
        states[uuid] = state


def get_state(message):
    uuid = message.from_user.id
    if uuid in states:
        return states[uuid]
    return 1


@bot.message_handler(commands=['upload'])
def set_upload_state(message):
    set_state(message, 0)
    bot.send_message(message.chat.id, "Отправьте боту изображения для пополнения базы данных утекших изображений")


@bot.message_handler(commands=['search'])
def set_search_state(message):
    set_state(message, 1)
    bot.send_message(message.chat.id, "Отправьте боту изображения для поиска по базе данных утекших изображений")


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Команды для работы с базой данных утекших изображений:\n" +
                                      "/upload - Пополнение базы данных\n/search - Поиск по базе данных")


@bot.message_handler(content_types=['photo'])
def image_match(message):
    print('We got new message containing photo')
    print(message)

    highest = 0
    highest_id = 0
    k = 0
    for photo in message.photo:
        if photo.height > highest:
            highest = photo.height
            highest_id = k
        k += 1

    highest_photo = message.photo[highest_id]

    file = bot.get_file(highest_photo.file_id)
    download_url = get_file_download_url(file.file_path)
    print('file can be downloaded at', download_url)
    print(file)

    if get_state(message) == 0:
        folder_to_save = 'pics'
    else:
        folder_to_save = 'temp'

    os.makedirs(folder_to_save, exist_ok=True)
    file_path = folder_to_save+'/'+file.file_id+'.jpg'
    urllib.request.urlretrieve(download_url, file_path)

    if get_state(message) == 0:
        bot.reply_to(message, "Ваше изображение успешно добавлено в базу данных")
    else:
        (result, db_pic) = is_file_in_folder(file_path)
        print('result', result)
        print('db_pic', db_pic)

        if result:
            bot.reply_to(message, "По вашему изображению нашлось совпадение в базе данных утекших фотографий")
            bot.send_photo(message.chat.id, db_pic.replace('.jpg', ''))
        else:
            bot.reply_to(message, "По вашему изображению не нашлось совпадений")

        os.remove(file_path)

    if get_state(message) == 0:
        bot.send_message(message.chat.id, "<i>*включен режим пополнения базы данных*</i>", parse_mode='HTML')
    else:
        bot.send_message(message.chat.id, "<i>*включен режим поиска изображений*</i>", parse_mode='HTML')


print('start polling ...')
bot.infinity_polling()
