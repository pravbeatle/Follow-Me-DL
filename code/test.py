image_dir = "../data/train/masks"

files_count = 0
hero_count = 0

os.chdir(image_dir)

for image in glob.glob('*.png'):
  files_count +=1
  img = misc.imread(image, flatten=False, mode='RGB')
  blue = img[:,:,2]

  if np.any(blue == 255):
    hero_count += 1

hero_percentage = hero_count / files_count * 100.

print (hero_percentage, "Percentage of images that contain the hero")


def flip_images(image_dir, extention):
	os.chdir(image_dir)

	images = glob.glob('*.' + extention)

	for index, image in enumerate(images):

		pic = misc.imread(image, flatten=False, mode = 'RGB')
		flipped_pic = np.fliplr(pic)

		misc.imsave('flipped' + image, flipped_pic)



flip_images('../train/masks/', 'png')
flip_images('../train/images/', 'jpeg')