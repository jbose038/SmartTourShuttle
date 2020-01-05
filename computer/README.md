<h3>About the files</h3>

<code>collect_training_data</code> : 수동 주행으로 데이터 수집

<code>mlp_training</code> : 수집된 데이터로 학습

<code>picam_calibration.py</code> : 캠 캘리브레이션 조정

<code>rc_driver.py</code> : 라즈베리파이(클라이언트)와 통신하며 명령 전달



<h4>PC(서버) 실행 파일</h4>

<code>rc_driver.py</code> : 먼저 실행하여 TCP 연결 대기시킨 후, 라즈베리파이에서 3개 파일 실행시켜 연결

