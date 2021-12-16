# 이 책에 기여하기
:label:`sec_how_to_contribute`

[readers](https://github.com/d2l-ai/d2l-en/graphs/contributors)의 기여는 이 책을 개선하는 데 도움이 됩니다.오타, 오래된 링크, 인용을 놓친 것, 코드가 우아하게 보이지 않거나 설명이 명확하지 않은 부분을 발견하면 다시 기여하고 독자를 도울 수 있도록 도와주세요.일반 서적에서는 인쇄 실행 간 (따라서 오타 수정 사이의 지연) 을 몇 년 단위로 측정 할 수 있지만 일반적으로 이 책의 개선 사항을 통합하는 데 몇 시간에서 며칠이 걸립니다.이 모든 것은 버전 관리 및 지속적인 통합 테스트로 인해 가능합니다.이렇게 하려면 깃허브 리포지토리에 [pull request](https://github.com/d2l-ai/d2l-en/pulls)를 제출해야 합니다.작성자가 풀 리퀘스트를 코드 저장소에 병합하면 기여자가 됩니다. 

## 마이너 텍스트 변경

가장 일반적인 기여는 한 문장을 편집하거나 오타를 수정하는 것입니다.[github repo](https732293614)에서 소스 파일을 찾아 마크다운 파일인 소스 파일을 찾는 것이 좋습니다.그런 다음 오른쪽 상단의 “이 파일 편집” 버튼을 클릭하여 마크다운 파일을 변경합니다. 

![Edit the file on Github.](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

완료되면 페이지 하단의 “파일 변경 제안” 패널에 변경 설명을 입력한 다음 “파일 변경 제안” 버튼을 클릭합니다.변경 사항을 검토할 수 있는 새 페이지로 리디렉션됩니다 (:numref:`fig_git_createpr`).모든 것이 정상이면 “풀 리퀘스트 생성” 버튼을 클릭하여 풀 리퀘스트를 제출할 수 있습니다. 

## 주요 변경 제안

텍스트나 코드의 상당 부분을 업데이트하려는 경우 이 책에서 사용하는 형식에 대해 조금 더 알아야 합니다.원본 파일은 방정식, 이미지, 장 및 인용 참조와 같이 [d2lbook](http://book.d2l.ai/user/markdown.html) 패키지를 통해 확장자가 설정된 [markdown format](https://daringfireball.net/projects/markdown/syntax)를 기반으로 합니다.마크다운 편집기를 사용하여 이러한 파일을 열고 변경할 수 있습니다. 

코드를 변경하려면 :numref:`sec_jupyter`에 설명된 대로 Jupyter를 사용하여 이러한 마크다운 파일을 여는 것이 좋습니다.따라서 변경 사항을 실행하고 테스트할 수 있습니다.변경 사항을 제출하기 전에 모든 출력을 지우는 것을 잊지 마십시오. CI 시스템은 업데이트한 섹션을 실행하여 출력을 생성합니다. 

일부 섹션은 여러 프레임워크 구현을 지원할 수 있으며, `d2lbook`를 사용하여 특정 프레임워크를 활성화할 수 있습니다. 따라서 다른 프레임워크 구현은 마크다운 코드 블록이 되며 Jupyter에서 “모두 실행” 시 실행되지 않습니다.즉, 먼저 를 실행하여 `d2lbook`를 설치합니다.

```bash
pip install git+https://github.com/d2l-ai/d2l-book
```

그런 다음 `d2l-en`의 루트 디렉터리에서 다음 명령 중 하나를 실행하여 특정 구현을 활성화할 수 있습니다.

```bash
d2lbook activate mxnet chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate pytorch chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate tensorflow chapter_multilayer-perceptrons/mlp-scratch.md
```

변경 사항을 제출하기 전에 모든 코드 블록 출력을 지우고 다음을 통해 모두 활성화하십시오.

```bash
d2lbook activate all chapter_multilayer-perceptrons/mlp-scratch.md
```

기본 구현이 아닌 새 코드 블록 (MXNet) 을 추가하는 경우 `# @tab` to mark this block on the beginning line. For example, ` # @tab pytorch` for a PyTorch code block, `# @tab 텐서플로` for a TensorFlow code block, or `# 텐서플로` for a TensorFlow code block, or `# all` 모든 구현에 대해 공유 코드 블록을 사용하십시오. @tab자세한 내용은 [d2lbook](http://book.d2l.ai/user/code_tabs.html)을 참조하십시오. 

## 새 섹션 또는 새 프레임워크 구현 추가

강화 학습과 같은 새로운 장을 만들거나 TensorFlow와 같은 새로운 프레임 워크의 구현을 추가하려면 먼저 이메일을 보내거나 [github issues](https://github.com/d2l-ai/d2l-en/issues)를 사용하여 작성자에게 문의하십시오. 

## 주요 변경 사항 제출

표준 `git` 프로세스를 사용하여 주요 변경 사항을 제출하는 것이 좋습니다.간단히 말해서 프로세스는 :numref:`fig_contribute`에 설명된 대로 작동합니다. 

![Contributing to the book.](../img/contribute.svg)
:label:`fig_contribute`

단계를 자세히 안내해 드리겠습니다.이미 Git에 익숙하다면 이 섹션을 건너뛸 수 있습니다.구체적으로 설명하면 기여자의 사용자 이름이 “astonzhang”이라고 가정합니다. 

### Git 설치하기

Git 오픈 소스 북은 [how to install Git](https://git-scm.com/book/en/v2)에 대해 설명합니다.이것은 일반적으로 우분투 리눅스에서 `apt install git`를 통해, 맥OS에 엑코드 개발자 도구를 설치하거나, 깃허브의 [desktop client](https://desktop.github.com)을 사용하여 작동합니다.GitHub 계정이 없는 경우 계정을 등록해야 합니다. 

### 깃허브에 로그인

브라우저에 책의 코드 리포지토리의 [address](https://github.com/d2l-ai/d2l-en/)을 입력합니다.:numref:`fig_git_fork`의 오른쪽 위에 있는 빨간색 상자에 있는 `Fork` 버튼을 클릭하여 이 책의 저장소를 복사합니다.이제*귀하의 사본*이 되었으며 원하는 방식으로 변경할 수 있습니다. 

![The code repository page.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`

이제 이 책의 코드 저장소는 스크린샷 :numref:`fig_git_forked`의 왼쪽 상단에 표시된 `astonzhang/d2l-en`와 같이 사용자 이름으로 포크됩니다 (즉, 복사). 

![Fork the code repository.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### 리포지토리 복제

저장소를 복제하려면 (즉, 로컬 복사본을 만들려면) 저장소 주소를 가져와야 합니다.:numref:`fig_git_clone`의 녹색 버튼은 이를 표시합니다.이 포크를 더 오래 유지하기로 결정한 경우 로컬 복사본이 메인 리포지토리에 최신 상태인지 확인하십시오.지금은 :ref:`chap_installation`의 지침을 따르기만 하면 됩니다.가장 큰 차이점은 이제 저장소의*자신의 포크*를 다운로드하고 있다는 것입니다. 

![Git clone.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-en.git
```

### 책 편집 및 푸시

이제 책을 편집할 차례입니다.:numref:`sec_jupyter`의 지침에 따라 목성에서 노트북을 편집하는 것이 가장 좋습니다.원하는 대로 변경하고 문제가 없는지 확인합니다.`~/d2l-en/챕터_부록_도구/하우투기여.md` 파일에서 오타를 수정했다고 가정합니다.그런 다음 변경한 파일을 확인할 수 있습니다. 

이 시점에서 Git은 `chapter_appendix_tools/how-to-contribute.md` 파일이 수정되었다는 메시지를 표시합니다.

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix_tools/how-to-contribute.md
```

원하는 것이 맞는지 확인한 후 다음 명령을 실행합니다.

```
git add chapter_appendix_tools/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```

변경된 코드는 저장소의 개인 포크에 저장됩니다.변경 사항 추가를 요청하려면 책의 공식 저장소에 대한 풀 리퀘스트를 만들어야 합니다. 

### 풀 리퀘스트

:numref:`fig_git_newpr`에서 볼 수 있듯이 GitHub의 리포지토리 포크로 이동하여 “새 풀 리퀘스트”를 선택합니다.그러면 편집 내용과 책의 기본 저장소에 있는 현재 내용 사이의 변경 사항을 보여주는 화면이 열립니다. 

![Pull Request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`

### 풀 리퀘스트 제출

마지막으로 :numref:`fig_git_createpr`에 표시된 버튼을 클릭하여 풀 리퀘스트를 제출합니다.풀 리퀘스트에 적용한 변경 사항을 설명해야 합니다.이렇게 하면 저자가 책을 더 쉽게 검토하고 책과 병합할 수 있습니다.변경 사항에 따라 즉시 수락되거나 거부되거나 변경 사항에 대한 피드백을 얻을 수 있습니다.일단 통합하면 갈 수 있습니다. 

![Create Pull Request.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`

풀 리퀘스트는 메인 저장소의 리퀘스트 목록에 나타날 것이다.신속하게 처리하기 위해 최선을 다할 것입니다. 

## 요약

* GitHub를 사용하여 이 책에 기여할 수 있습니다.
* 사소한 변경을 위해 GitHub에서 직접 파일을 편집할 수 있습니다.
* 주요 변경 사항이 있는 경우 리포지토리를 포크하고 로컬에서 편집한 다음 준비가 된 후에만 다시 기여하십시오.
* 풀 리퀘스트는 기여가 번들로 제공되는 방식입니다.거대한 풀 리퀘스트를 제출하지 마십시오. 이렇게 하면 이해하고 통합하기가 어려워집니다.작은 것을 여러 개 보내는 것이 좋습니다.

## 연습문제

1. `d2l-en` 저장소에 별표를 달고 포크하십시오.
1. 개선이 필요한 코드를 찾아 풀 리퀘스트를 제출하세요.
1. 우리가 놓친 레퍼런스를 찾아 풀 리퀘스트를 제출합니다.
1. 일반적으로 새 브랜치를 사용하여 풀 리퀘스트를 생성하는 것이 더 좋습니다.[Git branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)를 사용하여 이 작업을 수행하는 방법에 대해 알아보십시오.

[Discussions](https://discuss.d2l.ai/t/426)
