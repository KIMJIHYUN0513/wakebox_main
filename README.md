# wakebox_main
2019년 2학기 종합설계 프로젝트 Wake Box

### **프로젝트 소개** <br>
졸음방지를 위한 다양한 기능 및 스마트폰 앱과 연동하여 사용하는 블랙박스

### **주요 기능**
  1. 눈 깜빡임 및 고개숙임 인식<br>
  라즈베리파이 카메라를 이용해 실시간으로 차량 내외부 영상 촬영
  openCV, Dlib 라이브러리를 사용하여 내부 영상 속 얼굴인식 진행<br>
  1차로 동공인식, 2차로 고개숙임인식을 하여 설정해 둔 임계값을 초과하면 졸음인식으로 판단하여 경고음 발생
  
  2. 차량 내부 이산화탄소 측정<br>
  블랙박스 본체에 이산화탄소 센서를 부착하여 실시간으로 차량 내부 이산화탄소 측정<br>
  설정해 둔 임계값을 초과했을 경우 차량 환기 유도 경고음 발생
  
  3. 스마트폰 어플리케이션과 연동<br>
  블랙박스로 촬영한 차량 내외부 영상을 스마트폰 앱을 통해 실시간으로 확인<br>
  환경설정 기능을 추가하여 사용자에 따라 다양한 경고음 설정 가능
