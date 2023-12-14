const loginForm = document.getElementById("login-form");
const loginButton = document.getElementById("login-form-submit");
console.log( loginButton )
const loginErrorMsg = document.getElementById("login-error-msg");
const signForm = document.getElementById("sign-form");
const signupButton = document.getElementById("sign-form-submit");

// 임시
$('#username-field').val('test')
$('#password-field').val('123')



signupButton.addEventListener("click", (e) => {
    e.preventDefault();

    // Get user input
    const username = signForm.newUsername.value;
    const password = signForm.newPassword.value;
    const passwordConfirm = signForm.confirmPassword.value;

    // pw 일치여부 확인
    if (password != passwordConfirm) {
        alert("Passwords do not match.");
        return;
    }

    // id, pw 길이 제한
    if (username.length < 1 || password.length < 8) {
        alert("Username must be at least 5 characters long, and password must be at least 8 characters long.");
        return;
    }

    // 사용자 정보 생성
    const newUser = {
        username: username,
        password: password
    };

    // json 파일 접근
    fetch('_user_info.json')
        .then(response => response.json())
        .then(users => {
            // 사용자 정보 푸시
            users.push(newUser);

            // Write the updated user data back to the _user_info.json file
            const updatedUserData = JSON.stringify(users, null, 2);
            return fetch('_user_info.json', {
                method: 'POST', // You might need to use 'POST' or 'PATCH' depending on your server setup
                headers: {
                    'Content-Type': 'application/json'
                },
                body: updatedUserData
            });
        })
        .then(response => {
            if (response.ok) {
                alert("User registered successfully!");
                // Reset the form or perform any other necessary actions
                signForm.reset();
            } else {
                alert("Error registering user.");
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });

    // Reset the form or perform any other necessary actions
    signForm.reset();

})
    

loginForm.addEventListener("submit", function (event) {
    // 기본 폼 제출 방지
    event.preventDefault();

    // 사용자 입력 가져오기
    const username = loginForm.userid.value;
    const password = loginForm.password.value;
    console.log(username)
    console.log(password)
    // // jQuery를 사용하는 경우 서버에 POST 요청 보내기
    $.ajax({
        type: "POST",
        url: '/',
        data: { uid: username, upw: password },
        success: function (res) {
            console.log(res);
            let access_token = res.token;

            if (res.code == 1) {
                // 토큰을 '/' 경로에 쿠키에 저장
                $.cookie('token', access_token, { path: '/' });
                // Flask를 사용하는 경우 'chatbot' 페이지로 리다이렉트
                window.location.replace("/chatbot");
            } else {
                alert(res.msg); // 토큰 발급 실패
            }
        }
    });
    return false;
});

anime.timeline({loop: true})
  .add({
    targets: '.ml8 .circle-white',
    scale: [0, 3],
    opacity: [1, 0],
    easing: "easeInOutExpo",
    rotateZ: 360,
    duration: 1100
  }).add({
    targets: '.ml8 .circle-container',
    scale: [0, 1],
    duration: 1100,
    easing: "easeInOutExpo",
    offset: '-=1000'
  }).add({
    targets: '.ml8 .circle-dark',
    scale: [0, 1],
    duration: 1100,
    easing: "easeOutExpo",
    offset: '-=600'
  }).add({
    targets: '.ml8 .letters-left',
    scale: [0, 1],
    duration: 1200,
    offset: '-=550'
  }).add({
    targets: '.ml8 .bang',
    scale: [0, 1],
    rotateZ: [45, 15],
    duration: 1200,
    offset: '-=1000'
  }).add({
    targets: '.ml8',
    opacity: 0,
    duration: 1000,
    easing: "easeOutExpo",
    delay: 1400
  });

anime({
  targets: '.ml8 .circle-dark-dashed',
  rotateZ: 360,
  duration: 8000,
  easing: "linear",
  loop: true
});