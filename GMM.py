import numpy as np
import cv2

# Constants
INITIAL_VARIANCE = 400
LEAST_PROBABLE_VARIANCE = 200
LEAST_PROBABLE_OMEGA = 0.1
ALPHA = 0.2
THRESHOLD = 0.7

def norm_pdf(x, mean, sigma):
    return (1/(np.sqrt(2*3.14)*sigma))*(np.exp(-0.5*(((x-mean)/sigma)**2)))

def update_gaussian(frame_gray, gauss_fit_index, gauss_not_fit_index, mean, variance, omega):
    rho = ALPHA * norm_pdf(frame_gray[gauss_fit_index], mean[gauss_fit_index], np.sqrt(variance[gauss_fit_index]))
    constant = rho * ((frame_gray[gauss_fit_index] - mean[gauss_fit_index]) ** 2)
    mean[gauss_fit_index] = (1 - rho) * mean[gauss_fit_index] + rho * frame_gray[gauss_fit_index]
    variance[gauss_fit_index] = (1 - rho) * variance[gauss_fit_index] + constant
    omega[gauss_fit_index] = (1 - ALPHA) * omega[gauss_fit_index] + ALPHA
    omega[gauss_not_fit_index] = (1 - ALPHA) * omega[gauss_not_fit_index]
    return mean, variance, omega

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(r"Media/Original Video.mpg")
_,frame = cap.read()
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

row,col = frame.shape

mean = np.zeros([3,row,col],np.float64)
mean[1,:,:] = frame

variance = np.zeros([3,row,col],np.float64)
variance[:,:,:] = INITIAL_VARIANCE

omega = np.zeros([3,row,col],np.float64)
omega[0,:,:],omega[1,:,:],omega[2,:,:] = 0,0,1

omega_by_sigma = np.zeros([3,row,col],np.float64)

foreground = np.zeros([row,col],np.uint8)
background = np.zeros([row,col],np.uint8)

a = np.uint8([255])
b = np.uint8([0])

while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        print("Failed to capture frame")
        break

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray = frame_gray.astype(np.float64)

    variance[0][np.where(variance[0]<1)] = 10
    variance[1][np.where(variance[1]<1)] = 5
    variance[2][np.where(variance[2]<1)] = 1

    sigma1 = np.sqrt(variance[0])
    sigma2 = np.sqrt(variance[1])
    sigma3 = np.sqrt(variance[2])

    compare_val_1 = cv2.absdiff(frame_gray,mean[0])
    compare_val_2 = cv2.absdiff(frame_gray,mean[1])
    compare_val_3 = cv2.absdiff(frame_gray,mean[2])

    value1 = 2.5 * sigma1
    value2 = 2.5 * sigma2
    value3 = 2.5 * sigma3

    fore_index1 = np.where(omega[2]>THRESHOLD)
    fore_index2 = np.where(((omega[2]+omega[1])>THRESHOLD) & (omega[2]<THRESHOLD))

    gauss_fit_index1 = np.where(compare_val_1 <= value1)
    gauss_not_fit_index1 = np.where(compare_val_1 > value1)

    gauss_fit_index2 = np.where(compare_val_2 <= value2)
    gauss_not_fit_index2 = np.where(compare_val_2 > value2)

    gauss_fit_index3 = np.where(compare_val_3 <= value3)
    gauss_not_fit_index3 = np.where(compare_val_3 > value3)

    temp = np.zeros([row, col])
    temp[fore_index1] = 1
    temp[gauss_fit_index3] = temp[gauss_fit_index3] + 1
    index3 = np.where(temp == 2)

    temp = np.zeros([row,col])
    temp[fore_index2] = 1
    index = np.where((compare_val_3<=value3)|(compare_val_2<=value2))
    temp[index] = temp[index]+1
    index2 = np.where(temp==2)

    match_index = np.zeros([row,col])
    match_index[gauss_fit_index1] = 1
    match_index[gauss_fit_index2] = 1
    match_index[gauss_fit_index3] = 1
    not_match_index = np.where(match_index == 0)

    mean[0], variance[0], omega[0] = update_gaussian(frame_gray, gauss_fit_index1, gauss_not_fit_index1, mean[0], variance[0], omega[0])
    mean[1], variance[1], omega[1] = update_gaussian(frame_gray, gauss_fit_index2, gauss_not_fit_index2, mean[1], variance[1], omega[1])
    mean[2], variance[2], omega[2] = update_gaussian(frame_gray, gauss_fit_index3, gauss_not_fit_index3, mean[2], variance[2], omega[2])

    mean[0][not_match_index] = frame_gray[not_match_index]
    variance[0][not_match_index] = LEAST_PROBABLE_VARIANCE
    omega[0][not_match_index] = LEAST_PROBABLE_OMEGA

    sum = np.sum(omega,axis=0)
    omega = omega/sum

    omega_by_sigma[0] = omega[0] / sigma1
    omega_by_sigma[1] = omega[1] / sigma2
    omega_by_sigma[2] = omega[2] / sigma3

    index = np.argsort(omega_by_sigma,axis=0)

    mean = np.take_along_axis(mean,index,axis=0)
    variance = np.take_along_axis(variance,index,axis=0)
    omega = np.take_along_axis(omega,index,axis=0)

    frame_gray = frame_gray.astype(np.uint8)

    background[index2] = frame_gray[index2]
    background[index3] = frame_gray[index3]
    cv2.imshow('frame',cv2.subtract(frame_gray,background))
    cv2.imshow('frame_gray',frame_gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
