# Installation
### Windows
```bash
install.bat
```
### Ubuntu
```bash
bash install.sh
```
### Macos
```bash
bash install_osx.sh
```

# Data Preparation
- Set the value ***database.path*** in ***vnpy/trader/setting.py*** to a directory that you want to place the tick database
- Generate a VNPY format database:
    ```bash
    python hx_future_database/parse_hx_data_to_db.py --data_file_dir path_to_sh_data_dir --exchange SHFE
    python hx_future_database/parse_hx_data_to_db.py --data_file_dir path_to_cffex_data_dir --exchange CFFEX
    ```
