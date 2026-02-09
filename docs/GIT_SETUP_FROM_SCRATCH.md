# GitHub 저장소 연결부터 푸시까지 (처음부터)

아래 순서대로 하면 됩니다. 이미 한 단계를 했다면 그 다음부터 진행하면 됩니다.

---

## 0. 준비

- **GitHub 계정**: [github.com](https://github.com) 에서 가입
- **Git 설치**: 터미널에서 `git --version` 입력해 보기. 없으면 [git-scm.com](https://git-scm.com) 에서 설치
- **프로젝트 위치**: `~/Desktop/Spatial_MAS` (바탕화면의 Spatial_MAS 폴더)

---

## 1. GitHub에서 새 저장소(Repo) 만들기

1. 브라우저에서 **https://github.com** 접속 후 로그인
2. 오른쪽 위 **+** 클릭 → **New repository** 선택
3. 다음만 입력:
   - **Repository name**: `Spatial_MAS` (원하면 다른 이름도 가능)
   - **Public** 선택
   - **Add a README file**, **Add .gitignore**, **Choose a license** 는 **전부 체크하지 말고** 비워 둠
4. **Create repository** 클릭
5. 생성된 페이지에 **저장소 주소**가 나옴. 예:
   - HTTPS: `https://github.com/내아이디/Spatial_MAS.git`
   - SSH: `git@github.com:내아이디/Spatial_MAS.git`  
   → 이 주소를 복사해 둠 (아래 3단계에서 씀)

---

## 2. 로컬 폴더를 Git 저장소로 만들기

맥 기준, **터미널**을 열고 아래를 **한 줄씩** 실행:

```bash
cd ~/Desktop/Spatial_MAS
```

```bash
git init
```

- `Initialized empty Git repository in ...` 라고 나오면 성공.
- 이제 이 폴더가 “Git으로 관리되는 저장소”가 됨.

---

## 3. GitHub 저장소를 “원격(remote)”으로 연결하기

**2단계에서 복사한 주소**를 아래 `주소` 자리에 넣어서 실행.

**HTTPS 사용할 때 (비밀번호 대신 토큰 쓰는 방식):**
```bash
git remote add origin https://github.com/내아이디/Spatial_MAS.git
```
→ `내아이디`를 본인 GitHub 사용자명으로 바꾸기.

**SSH 사용할 때 (키 설정해 두었다면):**
```bash
git remote add origin git@github.com:내아이디/Spatial_MAS.git
```

연결 확인:
```bash
git remote -v
```
- `origin` 에 방금 넣은 주소가 두 줄로 나오면 됨.

---

## 4. 파일 올리기(커밋) & GitHub에 푸시하기

**4-1. 올릴 파일 선택**
```bash
git add .
```
- `.` 는 “현재 폴더 전체”를 스테이징.  
- `results/` 같은 건 `.gitignore` 에 있어서 자동으로 제외됨.

**4-2. 상태 확인 (선택)**
```bash
git status
```
- 초록색으로 나오는 파일들이 이번에 커밋될 대상.

**4-3. 첫 커밋 만들기**
```bash
git commit -m "Initial: STVQA-7K eval and failure analysis"
```
- `m "..."` 안의 메시지는 원하는 대로 바꿔도 됨.

**4-4. 기본 브랜치 이름을 main 으로 (필요할 때만)**
```bash
git branch -M main
```
- 이미 main 이면 그대로 두면 됨.

**4-5. GitHub에 푸시**
```bash
git push -u origin main
```

- **HTTPS** 로 했는데 로그인 창이 뜨면:
  - GitHub **사용자명** + **비밀번호** 대신 **Personal Access Token** 입력.
  - 토큰 만들기: GitHub → 우측 상단 프로필 → **Settings** → 왼쪽 맨 아래 **Developer settings** → **Personal access tokens** → **Generate new token**.  
    권한에 `repo` 체크 후 생성한 토큰을 복사해 비밀번호 자리에 붙여넣기.
- **SSH** 로 했는데 권한 오류가 나오면: SSH 키를 GitHub에 등록했는지 확인.

푸시가 끝나면 브라우저에서 `https://github.com/내아이디/Spatial_MAS` 를 새로고침하면 파일들이 보입니다.

---

## 5. 이후에 코드 수정했을 때 푸시하는 방법

수정한 뒤, 같은 폴더에서:

```bash
cd ~/Desktop/Spatial_MAS
git add .
git status
git commit -m "수정 내용 한 줄 요약"
git push
```

- 첫 푸시에서 `-u origin main` 을 했으면, 이후에는 `git push` 만 해도 됨.

---

## 6. 정리: 한 번에 복사해서 쓸 수 있는 명령어

**저장소를 방금 만들었고, 아직 로컬에서 `git init` 안 했을 때:**

```bash
cd ~/Desktop/Spatial_MAS
git init
git remote add origin https://github.com/내아이디/Spatial_MAS.git
git add .
git commit -m "Initial: STVQA-7K eval and failure analysis"
git branch -M main
git push -u origin main
```

→ **`내아이디`** 만 본인 GitHub 아이디로 바꾸면 됨.

---

## 자주 나오는 상황

| 상황 | 해결 |
|------|------|
| `git: command not found` | Git 설치: [git-scm.com](https://git-scm.com) |
| `remote origin already exists` | 이미 연결된 것. 푸시만 하면 됨: `git push -u origin main` |
| `Permission denied` (SSH) | SSH 키 생성 후 GitHub에 등록, 또는 HTTPS + 토큰 사용 |
| `Support for password authentication was removed` | 비밀번호 대신 **Personal Access Token** 사용 |

이렇게 하면 “Repo 연결하는 것부터” 처음부터 끝까지 한 번에 할 수 있습니다.
