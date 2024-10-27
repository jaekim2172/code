import cv2
import numpy as np

# 이미지 로드
image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')

# ORB 검출기 생성
orb = cv2.ORB_create()

# 이미지에서 특징점과 기술자(디스크립터) 추출
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# BFMatcher 생성 및 기술자 매칭
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 매칭 결과를 거리 기준으로 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 상위 매칭 10개 선택
matches = matches[:10]

# 매칭 결과를 이미지로 그리기
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 이미지 출력
cv2.imshow('Matched Features', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

