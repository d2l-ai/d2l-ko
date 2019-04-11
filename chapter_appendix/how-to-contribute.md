# 이 책에 기여하는 방법

이 오픈-소스 책의 기여자 목록을 [1]에서 볼 수 있습니다. 기여를 하고 싶다면, Git을 설치하고 이 책의 GitHub 코드 리포지토리에 pull request을 제출해주세요. 저자가 여러분의 pull request를 코드 리포지토리에 머지하면, 여러분은 기여자가 됩니다.

이 절에서는 이 책에 기여를 하는데 사용되는 기본적인 Git 절차를 설명하겠습니다. Git 사용법에 친숙하다면, 이 절을 넘어가도 됩니다.

아래 절차를 수행할 때, 기여자의 GitHub ID가 "astonzhang"이라고 가정하겠습니다.

1 단계: Git을 설치하세요. Git 오픈 소스 책[3]은 Git을 어떻게 설치하는지 상세하게 알려줍니다. 만약 아직 GitHub 계정이 없으면, 지금 가입하세요.[4]

2 단계: GitHub에 로그인 합니다. 웹 브라우저에 이 책의 코드 리포지토리 주소를 입력합니다.[2] 그림 11.20의 우측 상단에 빨간색으로 표시된 "Fork" 버튼을 클릭해서 이 책의 코드 리포지토리를 복제합니다.

![The code repository page.](../img/contrib01.png)

자 그럼 그림 11.21의 왼쪽 위에 보이는 "Your GitHub ID/d2l-ko"와 같이 여러분의 username에 이 책의 코드 리포지토리가 복사해졌을 것입니다. 

![Copy the code repository.](../img/contrib02.png)

3 단계: 그림 11.21의 오론쪽에 보이는 초록색 "Clone or download" 버튼을 클릭하고, 빨간색 박스로 표시한 버튼을 눌러서 여러분의 username 아래 있는 코드 리포지토리 주소를 복사합니다. ["Acquiring and Running Codes in This Book"](../chapter_prerequisite/install.md)를 참고해서 명령행 모드로 들어가세요. 여기서는 로컬 디스크의 "~/repo" 경로에 코드 리포지토리를 복사한다고 가정하겠습니다. 그럼 이 경로로 이동하고, `git clone`을 적고, 그 뒤에 여러분의 username이 포함된 코드 리포지토리의 주소를 붙여놓습니다. 이제 명령을 수행하세요.

```
# Replace your_Github_ID with your GitHub username
git clone https://github.com/your_Github_ID/d2l-ko.git
```

이제 이 책의 코드 리포지토리의 모든 파일이 로컬 디스크의 `~/repo/d2l-ko' 경로에 복사되었을 것입니다.

4 단계: 로컬 경로에 있는 코드 리포지토리를 수정하세요. `~/repo/d2l-ko/chapter_deep-learning-basics/linear-regression.md` 파일의 오타를 수정했다고 가정해보겠습니다. 명령행 모드에서, `~/repo/d2l-ko` 경로로 이동한 후 아래 명령을 수행하세요.

```
git status
```

그러면 그림 11.22에서 처럼 Git이 "chapter_deep-learning-basics/linear-regression.md" 파일이 수정되었다고 알려줍니다. 

![Git prompts that the chapter_deep-learning-basics/linear-regression.md file has been modified.](../img/contrib03.png)

변경을 제출할 파일이 정확하면, 아래 명령을 수행하세요.

```
git add chapter_deep-learning-basics/linear-regression.md
git commit -m 'fix typo in linear-regression.md'
git push
```

여기서 `'fix typo in linear-regression.md'` 는 제출하는 변경에 대한 설명입니다. 여러분이 제출하는 변경 내용에 따라서 바꿔주세요.

5 단계: 다시 웹 브라우저에서 이 책의 코드 리포지토리 주소[2]를 입력하세요. 그림 12.20의 좌측 하단의 빨간색 상자로 표시된 "New pull request" 버튼을 클릭하고, 이 후에 나오는 페이지에서 그럼 11.12의 오른쪽에 빨간색 박스로 표시된 "compare across forks" 링크를 클릭하세요. 다음으로 그 아래 빤간 박스로 표시된 "head fork:d2l-ai/d2l-ko"를 클릭합니다. 그럼 11.23에서 보이는 것처럼, 팝업 텍스트 상자에 여러분의 GitHub ID를 입력하고, 드롭 다운 메뉴에서 'YourGitHub-ID/d2l-ko'를 선택합니다.

![Select the code repository where the source of the change is located.](../img/contrib04.png)

6 단계: 그림 11.24 처럼, 제목과 본문 텍스트 상자에 여러분이 제출하는 pull request에 대한 설명을 적어주세요. 그리고, 녹색 "Create pull request" 버튼을 눌러서 pull request를 제출합니다.

![Describe and submit a pull request.](../img/contrib05.png)

요청이 제출되면, 그림 11.25과 같이 제출된 모든 pull request를 보여주는 페이지를 볼 수 있습니다.

![The pull request has been submitted.](../img/contrib06.png)

## 요약

* 여러분은 GitHub를 이용해서 이 책에 기여를 할 수 있습니다.

## 문제

* 이 책의 어떤 부분이 개선되어야 한다고 느낀다면, pull request를 제출해보세요.


## 참고 자료

[1] 이 책(한글판)의 기여자 목록. https://github.com/d2l-ai/d2l-ko/graphs/contributors

[2] 이 책의 코드 리포지토리 주소. https://github.com/d2l-ai/d2l-ko

[3] Git 설치하기. https://git-scm.com/book/zh/v2

[4] GitHub URL. https://github.com/

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2401)

![](../img/qr_how-to-contribute.svg)
