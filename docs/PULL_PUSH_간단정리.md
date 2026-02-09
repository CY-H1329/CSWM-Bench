# Pull / Push 간단 정리 (두 서버에서 같은 방식으로)

**저장소:** https://github.com/CY-H1329/Spatial_MAS

---

## 1. 중요한 것 하나

- **Git 명령어는 “Git 저장소 폴더” 안에서만 실행**해야 함.
- 그 폴더 안에는 `.git` 이 있어야 하고, 그곳에서만 `git pull` / `git push` 가 됨.

---

## 2. 서버에서 폴더가 겹쳐 있는 경우 (한 번만 정리)

지금처럼 `~/CY/Spatial_MAS`(Git 아님) 안에 `Spatial_MAS`(Git 있음)가 있으면,  
**바깥 폴더로 들어가서** `git pull` 해도 “not a git repository” 가 납니다.

**한 번만** 아래처럼 정리하면, 앞으로는 `~/CY/Spatial_MAS` 가 곧 Git 저장소가 됩니다.

```bash
cd ~/CY
mv Spatial_MAS Spatial_MAS_old
mv Spatial_MAS_old/Spatial_MAS Spatial_MAS
rmdir Spatial_MAS_old 2>/dev/null || true
```

이후부터는:

```bash
cd ~/CY/Spatial_MAS
git pull origin main
```

이렇게만 하면 됨.

---

## 3. 두 서버에서 쓰는 법 (같은 흐름)

| 어디서 | 할 일 | 명령어 |
|--------|--------|--------|
| **로컬(Mac)** | 프로젝트 폴더 | `~/Desktop/Spatial_MAS` |
| **서버(H100 등)** | 프로젝트 폴더 | `~/CY/Spatial_MAS` (위 정리 후) |

**최신 코드 받기 (Pull)**  
→ 항상 **프로젝트 폴더로 들어간 뒤** 실행:

```bash
cd 프로젝트폴더
git pull origin main
```

**수정 후 올리기 (Push)**  
→ 수정한 쪽에서:

```bash
cd 프로젝트폴더
git add .
git commit -m "무슨 수정 했는지 한 줄"
git push origin main
```

**다른 쪽에서 다시 받기**  
→ 로컬에서 push 했으면 서버에서 pull, 서버에서 push 했으면 로컬에서 pull:

```bash
cd 프로젝트폴더
git pull origin main
```

---

## 4. 요약

- **Pull/Push 할 때는 반드시 “Git이 있는 프로젝트 폴더”에서.**
- 로컬: `cd ~/Desktop/Spatial_MAS`  
  서버(정리 후): `cd ~/CY/Spatial_MAS`
- **받을 때:** `git pull origin main`  
  **올릴 때:** `git add .` → `git commit -m "메시지"` → `git push origin main`

이렇게만 하면 두 서버 모두 같은 방식으로 pull/push 할 수 있음.
