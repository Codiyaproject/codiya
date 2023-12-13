        // 서버 접속 url 획득 ( socket.io => http:// , websocket => ws:// )
        let url = `http://${ document.domain }:${ location.port }`
        console.log( url )
        // 서버 접속
        const socket = io(url)
        // 연결이 됬음을 알려주는 이벤트가 발생되면
        // 이벤트명 사전에 정의된것들,  사용자 정의 이벤트(서버와 약속된 통신을 진행-> TR, 통신프로토콜)
        socket.on('connect', ()=>{
            console.log('서버에 접속 되었다')

        })
        // 서버가 보내는 메시지를 받는 부분
        socket.on('sToc_send_msg', data=>{
            console.log(data)
            addMessage(data.sender, data.msg)

        })

        // 채팅창이 열리면 대화내용 삭제(깨긋하게 비움)
        // $('#chat_board').empty()
        // 채팅 UI 관련 이벤트 설정
        // 입력창
        $('#chat_input').on('keypress', e=>{            
            if( e.keyCode == 13){
                sendMessage()
            }
        })
        // SEND 버튼
        $('#chat_snd').on('click', function(e){            
            sendMessage()
        })
        // 사용자가 입력한 내용을 서버로 전송
        function sendMessage(){
            // 1. 메시지 획득
            msg = $('#chat_input').val()
            // 2. 입력창 비우기
            $('#chat_input').val('')
            // 3. 채팅창에 입력 내용 채우기
            addMessage( 'me', msg )
            // 4. 서버로 소켓을 통해서 메세지를 전송한다
            //    이벤트명(서버와클라이언트간 서로 약속한 이벤트,통신프로토콜), 파라마터(전송데이터)
            //    js의 객체 {} => json => 파이썬 dict
            socket.emit('cTos_send_msg', {chatMsg:msg, dumy:'hello socket'})
        }
        // 채팅 보드에 채팅 내용 추가하기
        function addMessage( user='me', msg ){
            let html = null
            if( user === 'me'){
                // 메시지는 오른쪽에 붙어서 표기
                html = `
                <div class="ChatItem ChatItem--expert">
                <div class="ChatItem-meta">
                  <div class="ChatItem-avatar">
                    <img class="ChatItem-avatarImage" src="https://randomuser.me/api/portraits/women/0.jpg">
                  </div>
                </div>
                <div class="ChatItem-chatContent">
                  <div class="ChatItem-chatText">${msg}</div>
                  <div class="ChatItem-timeStamp"><strong>Me</strong> • Today ${new Date()}</div>
                </div>
              </div>             
                `
            }else{
                // 메시지는 왼쪽에 붙어서 표기
                html = `
                    <div class="direct-chat-msg">
                        <div class="direct-chat-infos clearfix">
                            <span class="direct-chat-name float-left">AI</span>
                            <span class="direct-chat-timestamp float-right">${new Date()}</span>
                        </div>

                        <img class="direct-chat-img" src="https://randomuser.me/api/portraits/women/0.jpg"
                            alt="message user image">

                        <div class="direct-chat-text">
                            ${msg}
                        </div>

                    </div>
                `
            }
            $('#chat_board').append( html )
            // 스크롤 처리
            $("#chat_board").scrollTop($("#chat_board")[0].scrollHeight)
        }