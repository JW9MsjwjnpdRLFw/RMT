import cv2
import argparse
import numpy as np
import os

def get_noise(img,value=10):  
    
    noise = np.random.uniform(0,256,img.shape[0:2])

    v = value *0.01
    noise[np.where(noise<(256-v))]=0
    
    

    k = np.array([ [0, 0.1, 0],
                    [0.1,  8, 0.1],
                    [0, 0.1, 0] ])
            
    noise = cv2.filter2D(noise,-1,k)
    

    '''cv2.imshow('img',noise)
    cv2.waitKey()
    cv2.destroyWindow('img')'''
    return noise

def rain_blur(noise, length=10, angle=0,w=1):

    trans = cv2.getRotationMatrix2D((length/2, length/2), angle-45, 1-length/100.0)  
    dig = np.diag(np.ones(length))   
    k = cv2.warpAffine(dig, trans, (length, length))  
    k = cv2.GaussianBlur(k,(w,w),0)    

    blurred = cv2.filter2D(noise, -1, k)    
    
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    '''
    cv2.imshow('img',blurred)
    cv2.waitKey()
    cv2.destroyWindow('img')'''
    
    return blurred

def alpha_rain(rain,img,beta = 0.8):
    
    rain = np.expand_dims(rain,2)
    rain_effect = np.concatenate((img,rain),axis=2)  #add alpha channel

    rain_result = img.copy()   
    rain = np.array(rain,dtype=np.float32)     
    rain_result[:,:,0]= rain_result[:,:,0] * (255-rain[:,:,0])/255.0 + beta*rain[:,:,0]
    rain_result[:,:,1] = rain_result[:,:,1] * (255-rain[:,:,0])/255 + beta*rain[:,:,0] 
    rain_result[:,:,2] = rain_result[:,:,2] * (255-rain[:,:,0])/255 + beta*rain[:,:,0]

    # cv2.imshow('rain_effct_result',rain_result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return rain_result

def add_rain(rain,img,alpha=0.9):

    #chage rain into  3-dimenis

    rain = np.expand_dims(rain,2)
    rain = np.repeat(rain,3,2)

    result = cv2.addWeighted(img,alpha,rain,1-alpha,1)
    cv2.imshow('rain_effct',result)
    cv2.waitKey()
    cv2.destroyWindow('rain_effct')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="../source_datasets/orginal")
    parser.add_argument('--output_path', type=str, default="../follow_up_datasets/rainy")

    args = parser.parse_args()
    print("Rainy config: {}".format(args))
    if not os.path.exists(os.path.join(args.output_path, 'source_datasets')):
        os.makedirs(os.path.join(args.output_path, 'source_datasets'))

    if not os.path.exists(os.path.join(args.output_path, 'follow_up_datasets')):
        os.makedirs(os.path.join(args.output_path, 'follow_up_datasets'))
        
    source_path = args.input_path
    img_list = os.listdir(source_path)
    for img_name in img_list:
        img = cv2.imread(os.path.join(source_path, img_name))
        cv2.imwrite(os.path.join(args.output_path, 'source_datasets', img_name), img)
        noise = get_noise(img,value=200)
        rain = rain_blur(noise,length=30,angle=-30,w=3)
        rain_img = alpha_rain(rain,img,beta=0.6)  #方法一，透明度賦值
        cv2.imwrite(os.path.join(args.output_path, 'follow_up_datasets', img_name), rain_img)